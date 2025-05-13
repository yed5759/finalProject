#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from piano_transformer import PianoTransformer, preprocess_audio, create_midi_from_predictions
import pretty_midi

class PianoDataset(Dataset):
    """Dataset for piano transcription"""
    
    def __init__(self, data_dir, split='train', feature_type='mel', 
                 sample_rate=16000, n_fft=2048, hop_length=512,
                 sequence_length=None, max_files=None):
        """
        Initialize dataset
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            feature_type: 'mel', 'cqt', or 'both'
            sample_rate: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            sequence_length: Sequence length (if None, use full sequences)
            max_files: Maximum number of files to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sequence_length = sequence_length
        
        # Get audio and MIDI paths
        audio_dir = self.data_dir / split / 'audio'
        midi_dir = self.data_dir / split / 'midi'
        
        # Find matching audio and MIDI files
        self.audio_files = sorted(list(audio_dir.glob('*.wav')))
        midi_files = {f.stem: f for f in midi_dir.glob('*.mid')}
        
        # Only keep files that have both audio and MIDI
        self.pairs = []
        for audio_file in self.audio_files:
            if audio_file.stem in midi_files:
                self.pairs.append({
                    'audio': audio_file,
                    'midi': midi_files[audio_file.stem]
                })
        
        # Limit number of files if specified
        if max_files is not None:
            self.pairs = self.pairs[:max_files]
            
        print(f"Loaded {len(self.pairs)} {split} examples")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Process audio
        features, _ = preprocess_audio(
            pair['audio'], 
            sample_rate=self.sample_rate,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            feature_type=self.feature_type
        )
        features = torch.tensor(features, dtype=torch.float32).T  # (time, features)
        
        # Extract piano roll from MIDI
        midi_data = pretty_midi.PrettyMIDI(str(pair['midi']))
        piano_roll = self._extract_piano_roll(midi_data, features.shape[0])
        
        # Apply sequence length if specified
        if self.sequence_length is not None and features.shape[0] > self.sequence_length:
            # Randomly select a sequence
            max_start = features.shape[0] - self.sequence_length
            start = np.random.randint(0, max_start)
            end = start + self.sequence_length
            
            features = features[start:end]
            piano_roll = piano_roll[start:end]
        
        return features, piano_roll
    
    def _extract_piano_roll(self, midi_data, length):
        """Extract piano roll from MIDI data"""
        # Time between frames
        hop_time = self.hop_length / self.sample_rate
        
        # Get piano roll (frame_rate = frames per second)
        frame_rate = 1 / hop_time
        piano_roll = midi_data.get_piano_roll(fs=frame_rate)
        
        # Limit to 88 keys (from A0 to C8)
        piano_roll = piano_roll[21:109, :]
        
        # Transpose to (time, pitch)
        piano_roll = piano_roll.T
        
        # Ensure piano roll has the same length as features
        if piano_roll.shape[0] < length:
            # Pad
            padding = np.zeros((length - piano_roll.shape[0], piano_roll.shape[1]))
            piano_roll = np.concatenate((piano_roll, padding), axis=0)
        elif piano_roll.shape[0] > length:
            # Truncate
            piano_roll = piano_roll[:length, :]
        
        # Binarize and convert to tensor
        piano_roll = (piano_roll > 0).astype(np.float32)
        return torch.tensor(piano_roll, dtype=torch.float32)

def train_model(args):
    """Train the piano transformer model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    input_dim = 128
    if args.feature_type == 'cqt':
        input_dim = 84 + 1  # CQT bins + onset
    elif args.feature_type == 'both':
        input_dim = 128 + 84 + 1  # Mel + CQT + onset
    elif args.feature_type == 'mel':
        input_dim = 128 + 1  # Mel + onset
    
    model = PianoTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        output_dim=88,
        dropout=args.dropout
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Create datasets and dataloaders
    train_dataset = PianoDataset(
        args.data_dir, 
        split='train',
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        n_fft=args.fft_size,
        hop_length=args.hop_length,
        sequence_length=args.sequence_length,
        max_files=args.max_files
    )
    
    val_dataset = PianoDataset(
        args.data_dir, 
        split='val',
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        n_fft=args.fft_size,
        hop_length=args.hop_length,
        max_files=args.max_files
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features, targets = features.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            
            # Calculate accuracy
            accuracy = ((outputs > 0.5) == (targets > 0.5)).float().mean().item()
            
            # Update statistics
            train_loss += loss.item()
            train_accuracy += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': accuracy
            })
            
            # Free up memory
            del features, targets, outputs
            torch.cuda.empty_cache()
        
        # Calculate average training statistics
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_time = time.time() - start_time
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation"):
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                accuracy = ((outputs > 0.5) == (targets > 0.5)).float().mean().item()
                
                # Update statistics
                val_loss += loss.item()
                val_accuracy += accuracy
                
                # Free up memory
                del features, targets, outputs
                torch.cuda.empty_cache()
        
        # Calculate average validation statistics
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_time = time.time() - start_time
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Time: {train_time:.2f}s - "
              f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, Time: {val_time:.2f}s")
        
        # Save model if it's the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / f"best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Regularly save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(output_dir / "loss_plot.png")
    
    # Save hyperparameters
    with open(output_dir / "hyperparameters.json", 'w') as f:
        json.dump({
            'feature_type': args.feature_type,
            'sample_rate': args.sample_rate,
            'fft_size': args.fft_size,
            'hop_length': args.hop_length,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'sequence_length': args.sequence_length,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'gradient_clip': args.gradient_clip,
            'best_val_loss': best_val_loss,
            'training_date': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train piano transformer model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Path to output directory')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to use (for debugging)')
    
    # Audio parameters
    parser.add_argument('--feature-type', type=str, default='mel', choices=['mel', 'cqt', 'both'],
                      help='Feature type for audio representation')
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050],
                      help='Audio sample rate')
    parser.add_argument('--fft-size', type=int, default=2048,
                      help='FFT window size')
    parser.add_argument('--hop-length', type=int, default=512,
                      help='Hop length for FFT')
    
    # Model parameters
    parser.add_argument('--d-model', type=int, default=512,
                      help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=2048,
                      help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--sequence-length', type=int, default=1024,
                      help='Sequence length for training')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                      help='Gradient clipping value')
    parser.add_argument('--save-every', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main() 
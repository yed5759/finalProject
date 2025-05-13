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

from piano_transformer import PianoTransformer, preprocess_audio
import pretty_midi

class PianoDataset(Dataset):
    """Dataset for piano transcription"""
    
    def __init__(self, data_dir, split='train', 
                 sample_rate=16000, n_fft=2048, hop_length=512, n_cqt_bins=84,
                 sequence_length=None, max_files=None):
        """
        Initialize dataset
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            sample_rate: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_cqt_bins: Number of CQT bins
            sequence_length: Sequence length (if None, use full sequences)
            max_files: Maximum number of files to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_cqt_bins = n_cqt_bins
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
        
        # Process audio to extract CQT and onset/offset/velocity features
        features, _ = preprocess_audio(
            pair['audio'], 
            sample_rate=self.sample_rate,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            n_cqt_bins=self.n_cqt_bins
        )
        features = torch.tensor(features, dtype=torch.float32).T  # (time, features)
        
        # Extract piano roll with onset, offset and velocity information
        midi_data = pretty_midi.PrettyMIDI(str(pair['midi']))
        piano_roll = self._extract_piano_roll_with_onsets_offsets_velocity(midi_data, features.shape[0])
        
        # Apply sequence length if specified
        if self.sequence_length is not None and features.shape[0] > self.sequence_length:
            # Randomly select a sequence
            max_start = features.shape[0] - self.sequence_length
            start = np.random.randint(0, max_start)
            end = start + self.sequence_length
            
            features = features[start:end]
            piano_roll = piano_roll[start:end]
        
        return features, piano_roll
    
    def _extract_piano_roll_with_onsets_offsets_velocity(self, midi_data, length):
        """
        Extract piano roll with onset, offset, and velocity information
        
        Args:
            midi_data: PrettyMIDI object
            length: Desired length of the piano roll
            
        Returns:
            piano_roll: Piano roll with shape (length, 88*3)
                First 88 columns: Note onsets
                Next 88 columns: Note offsets
                Last 88 columns: Note velocities
        """
        # Time between frames
        hop_time = self.hop_length / self.sample_rate
        
        # Create empty piano roll with 88*3 columns
        piano_roll = np.zeros((length, 88 * 3), dtype=np.float32)
        
        # Get all piano notes
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Skip drum tracks
                notes.extend(instrument.notes)
        
        # Sort notes by start time
        notes.sort(key=lambda x: x.start)
        
        # Fill in onset, offset, and velocity information
        for note in notes:
            # Only consider notes in the 88-key piano range (21-108)
            if not (21 <= note.pitch <= 108):
                continue
                
            # Convert MIDI pitch to piano key index (0-87)
            piano_key = note.pitch - 21
            
            # Convert start and end times to frame indices
            start_frame = int(note.start / hop_time)
            end_frame = int(note.end / hop_time)
            
            # Ensure frames are within bounds
            if start_frame >= length or end_frame < 0:
                continue
                
            start_frame = max(0, start_frame)
            end_frame = min(length - 1, end_frame)
            
            # Mark onset frame
            piano_roll[start_frame, piano_key] = 1.0  # Onset
            
            # Mark offset frame
            if end_frame < length:
                piano_roll[end_frame, piano_key + 88] = 1.0  # Offset
            
            # Fill in velocity for all active frames
            # Normalize velocity to [0, 1]
            normalized_velocity = note.velocity / 127.0
            piano_roll[start_frame:end_frame+1, piano_key + 2*88] = normalized_velocity
        
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
    input_dim = args.cqt_bins + 3  # CQT bins + onset + offset + velocity features
    
    model = PianoTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        output_dim=88 * 3,  # 88 piano keys * 3 outputs (onset, offset, velocity)
        dropout=args.dropout
    ).to(device)
    
    # Define loss function and optimizer
    # Use weighted BCE loss to balance the importance of onset, offset, and velocity prediction
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
        sample_rate=args.sample_rate,
        n_fft=args.fft_size,
        hop_length=args.hop_length,
        n_cqt_bins=args.cqt_bins,
        sequence_length=args.sequence_length,
        max_files=args.max_files
    )
    
    val_dataset = PianoDataset(
        args.data_dir, 
        split='val',
        sample_rate=args.sample_rate,
        n_fft=args.fft_size,
        hop_length=args.hop_length,
        n_cqt_bins=args.cqt_bins,
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
        train_onset_accuracy = 0.0
        train_offset_accuracy = 0.0
        train_velocity_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features, targets = features.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Split outputs and targets for separate metrics
            onset_output = outputs[:, :, :88]
            offset_output = outputs[:, :, 88:176]
            velocity_output = outputs[:, :, 176:264]
            
            onset_target = targets[:, :, :88]
            offset_target = targets[:, :, 88:176]
            velocity_target = targets[:, :, 176:264]
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            
            # Calculate accuracies
            onset_accuracy = ((onset_output > 0.5) == (onset_target > 0.5)).float().mean().item()
            offset_accuracy = ((offset_output > 0.5) == (offset_target > 0.5)).float().mean().item()
            velocity_loss_val = nn.MSELoss()(velocity_output, velocity_target).item()
            
            # Update statistics
            train_loss += loss.item()
            train_onset_accuracy += onset_accuracy
            train_offset_accuracy += offset_accuracy
            train_velocity_loss += velocity_loss_val
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'onset_acc': onset_accuracy,
                'offset_acc': offset_accuracy
            })
            
            # Free up memory
            del features, targets, outputs
            torch.cuda.empty_cache()
        
        # Calculate average training statistics
        train_loss /= len(train_loader)
        train_onset_accuracy /= len(train_loader)
        train_offset_accuracy /= len(train_loader)
        train_velocity_loss /= len(train_loader)
        train_time = time.time() - start_time
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_onset_accuracy = 0.0
        val_offset_accuracy = 0.0
        val_velocity_loss = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation"):
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Split outputs and targets for separate metrics
                onset_output = outputs[:, :, :88]
                offset_output = outputs[:, :, 88:176]
                velocity_output = outputs[:, :, 176:264]
                
                onset_target = targets[:, :, :88]
                offset_target = targets[:, :, 88:176]
                velocity_target = targets[:, :, 176:264]
                
                # Calculate accuracies
                onset_accuracy = ((onset_output > 0.5) == (onset_target > 0.5)).float().mean().item()
                offset_accuracy = ((offset_output > 0.5) == (offset_target > 0.5)).float().mean().item()
                velocity_loss_val = nn.MSELoss()(velocity_output, velocity_target).item()
                
                # Update statistics
                val_loss += loss.item()
                val_onset_accuracy += onset_accuracy
                val_offset_accuracy += offset_accuracy
                val_velocity_loss += velocity_loss_val
                
                # Free up memory
                del features, targets, outputs
                torch.cuda.empty_cache()
        
        # Calculate average validation statistics
        val_loss /= len(val_loader)
        val_onset_accuracy /= len(val_loader)
        val_offset_accuracy /= len(val_loader)
        val_velocity_loss /= len(val_loader)
        val_time = time.time() - start_time
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Onset Acc: {train_onset_accuracy:.4f}, Offset Acc: {train_offset_accuracy:.4f}, "
              f"Vel Loss: {train_velocity_loss:.4f}, Time: {train_time:.2f}s - "
              f"Val Loss: {val_loss:.4f}, Onset Acc: {val_onset_accuracy:.4f}, Offset Acc: {val_offset_accuracy:.4f}, "
              f"Vel Loss: {val_velocity_loss:.4f}, Time: {val_time:.2f}s")
        
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
            'sample_rate': args.sample_rate,
            'fft_size': args.fft_size,
            'hop_length': args.hop_length,
            'cqt_bins': args.cqt_bins,
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
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050],
                      help='Audio sample rate')
    parser.add_argument('--fft-size', type=int, default=2048,
                      help='FFT window size')
    parser.add_argument('--hop-length', type=int, default=512,
                      help='Hop length for FFT')
    parser.add_argument('--cqt-bins', type=int, default=84,
                      help='Number of CQT bins')
    
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
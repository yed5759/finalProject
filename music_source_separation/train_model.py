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

from piano_transformer import PianoTransformer, PianoTranscriptionDataset, collate_fn

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
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = PianoTransformer(
        n_cqt_bins=args.n_cqt_bins,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    print(model)
    print(f"Total parameters: {sum(p.nuget() for p in model.parameters() if p.requires_grad)}")
    
    # Create datasets
    train_dataset = PianoTranscriptionDataset(
        audio_dir=Path(args.data_dir) / 'train' / 'audio',
        midi_dir=Path(args.data_dir) / 'train' / 'midi',
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_cqt_bins=args.n_cqt_bins
    )
    
    val_dataset = PianoTranscriptionDataset(
        audio_dir=Path(args.data_dir) / 'val' / 'audio',
        midi_dir=Path(args.data_dir) / 'val' / 'midi',
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_cqt_bins=args.n_cqt_bins,
        random_offset=False  # No random offsets for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Define loss function for onset, offset, and velocity
    onset_loss_fn = torch.nn.BCEWithLogitsLoss()
    offset_loss_fn = torch.nn.BCEWithLogitsLoss()
    velocity_loss_fn = torch.nn.MSELoss()
    
    # Training and validation history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_onset_loss': [],
        'train_offset_loss': [],
        'train_velocity_loss': [],
        'val_onset_loss': [],
        'val_offset_loss': [],
        'val_velocity_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        train_losses = []
        train_onset_losses = []
        train_offset_losses = []
        train_velocity_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            audio_features, target_onsets, target_offsets, target_velocities = [x.to(device) for x in batch]
            
            # Forward pass
            pred_onsets, pred_offsets, pred_velocities = model(audio_features)
            
            # Calculate losses
            onset_loss = onset_loss_fn(pred_onsets, target_onsets)
            offset_loss = offset_loss_fn(pred_offsets, target_offsets)
            velocity_loss = velocity_loss_fn(pred_velocities * target_onsets, target_velocities * target_onsets)
            
            # Combined loss (weighted sum)
            loss = args.onset_weight * onset_loss + args.offset_weight * offset_loss + args.velocity_weight * velocity_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_onset_losses.append(onset_loss.item())
            train_offset_losses.append(offset_loss.item())
            train_velocity_losses.append(velocity_loss.item())
        
        train_loss = np.mean(train_losses)
        train_onset_loss = np.mean(train_onset_losses)
        train_offset_loss = np.mean(train_offset_losses)
        train_velocity_loss = np.mean(train_velocity_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_onset_losses = []
        val_offset_losses = []
        val_velocity_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                audio_features, target_onsets, target_offsets, target_velocities = [x.to(device) for x in batch]
                
                # Forward pass
                pred_onsets, pred_offsets, pred_velocities = model(audio_features)
                
                # Calculate losses
                onset_loss = onset_loss_fn(pred_onsets, target_onsets)
                offset_loss = offset_loss_fn(pred_offsets, target_offsets)
                velocity_loss = velocity_loss_fn(pred_velocities * target_onsets, target_velocities * target_onsets)
                
                # Combined loss
                loss = args.onset_weight * onset_loss + args.offset_weight * offset_loss + args.velocity_weight * velocity_loss
                
                # Record losses
                val_losses.append(loss.item())
                val_onset_losses.append(onset_loss.item())
                val_offset_losses.append(offset_loss.item())
                val_velocity_losses.append(velocity_loss.item())
        
        val_loss = np.mean(val_losses)
        val_onset_loss = np.mean(val_onset_losses)
        val_offset_loss = np.mean(val_offset_losses)
        val_velocity_loss = np.mean(val_velocity_losses)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_onset_loss'].append(train_onset_loss)
        history['train_offset_loss'].append(train_offset_loss)
        history['train_velocity_loss'].append(train_velocity_loss)
        history['val_onset_loss'].append(val_onset_loss)
        history['val_offset_loss'].append(val_offset_loss)
        history['val_velocity_loss'].append(val_velocity_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} (Onset: {train_onset_loss:.4f}, Offset: {train_offset_loss:.4f}, Velocity: {train_velocity_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Onset: {val_onset_loss:.4f}, Offset: {val_offset_loss:.4f}, Velocity: {val_velocity_loss:.4f})")
        print(f"Learning Rate: {current_lr}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}_loss_{val_loss:.4f}.pt"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model separately
            best_model_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        
        # Save latest model state
        latest_model_path = model_dir / "latest_model.pt"
        torch.save(model.state_dict(), latest_model_path)
        
        # Plot training curves (every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            plt.figure(figsize=(15, 10))
            
            # Plot combined loss
            plt.subplot(2, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Total Loss')
            plt.legend()
            
            # Plot onset loss
            plt.subplot(2, 2, 2)
            plt.plot(history['train_onset_loss'], label='Train Onset Loss')
            plt.plot(history['val_onset_loss'], label='Validation Onset Loss')
            plt.title('Onset Loss')
            plt.legend()
            
            # Plot offset loss
            plt.subplot(2, 2, 3)
            plt.plot(history['train_offset_loss'], label='Train Offset Loss')
            plt.plot(history['val_offset_loss'], label='Validation Offset Loss')
            plt.title('Offset Loss')
            plt.legend()
            
            # Plot velocity loss
            plt.subplot(2, 2, 4)
            plt.plot(history['train_velocity_loss'], label='Train Velocity Loss')
            plt.plot(history['val_velocity_loss'], label='Validation Velocity Loss')
            plt.title('Velocity Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(model_dir / f"training_curves_epoch_{epoch+1}.png")
            plt.close()
    
    # Save final model and training history
    final_model_path = model_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    
    history_path = model_dir / "training_history.pt"
    torch.save(history, history_path)
    
    print(f"\nTraining completed. Final model saved to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train piano transcription model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing prepared data')
    parser.add_argument('--model-dir', type=str, default='models/piano_transformer',
                        help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--n-cqt-bins', type=int, default=88,
                        help='Number of CQT bins')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of the model')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--segment-length', type=float, default=10.0,
                        help='Segment length in seconds for training')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training even if CUDA is available')
    
    # Loss weights
    parser.add_argument('--onset-weight', type=float, default=1.0,
                        help='Weight for onset loss')
    parser.add_argument('--offset-weight', type=float, default=0.5,
                        help='Weight for offset loss')
    parser.add_argument('--velocity-weight', type=float, default=0.3,
                        help='Weight for velocity loss')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main() 
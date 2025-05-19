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

def train_model(args):
    """Train the piano transformer model"""
    # Set device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
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
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
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
    # Use BCEWithLogitsLoss for binary classification tasks (onset, offset)
    # This combines sigmoid with BCE loss and handles numerical stability
    pos_weight_onset = torch.tensor([args.onset_pos_weight] * 88).to(device)
    pos_weight_offset = torch.tensor([args.offset_pos_weight] * 88).to(device)
    
    onset_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_onset)
    offset_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_offset)
    velocity_loss_fn = torch.nn.MSELoss()
    
    # Function to monitor prediction statistics during training
    def calculate_prediction_stats(logits_tensor, threshold=0.5):
        """Calculate statistics about model predictions to monitor training progress"""
        with torch.no_grad():
            # Apply sigmoid to convert logits to probabilities
            probs_tensor = torch.sigmoid(logits_tensor)
            binary_preds = (probs_tensor > threshold).float()
            stats = {
                'min': probs_tensor.min().item(),
                'max': probs_tensor.max().item(),
                'mean': probs_tensor.mean().item(),
                'median': probs_tensor.median().item(),
                'percent_positive': binary_preds.mean().item() * 100,
                'num_positive': binary_preds.sum().item()
            }
        return stats
    
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
        train_activation_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to device and ensure it's the right type
            audio_features, target_onsets, target_offsets, target_velocities = [x.to(device) for x in batch]
            
            # Forward pass
            onset_logits, offset_logits, velocity_logits = model(audio_features)
            
            # Calculate losses
            onset_loss = onset_loss_fn(onset_logits, target_onsets)
            offset_loss = offset_loss_fn(offset_logits, target_offsets)
            
            # For velocity, apply sigmoid to get probabilities, then calculate loss only for active notes
            velocity_probs = torch.sigmoid(velocity_logits)
            velocity_loss = velocity_loss_fn(velocity_probs * target_onsets, target_velocities * target_onsets)
            
            # Add activation target loss - encourages predictions to be more balanced
            # Use logit space for this calculation
            min_activation_target = args.min_activation_target
            activation_loss_onset = torch.nn.functional.mse_loss(
                torch.sigmoid(onset_logits).mean(), 
                torch.tensor(min_activation_target, device=device)
            )
            activation_loss_offset = torch.nn.functional.mse_loss(
                torch.sigmoid(offset_logits).mean(), 
                torch.tensor(min_activation_target, device=device)
            )
            activation_loss = activation_loss_onset + activation_loss_offset
            
            # Combined loss (weighted sum)
            loss = args.onset_weight * onset_loss + args.offset_weight * offset_loss + args.velocity_weight * velocity_loss
            
            # Add activation loss with its weight
            if args.activation_loss_weight > 0:
                loss = loss + args.activation_loss_weight * activation_loss
            
            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_onset_losses.append(onset_loss.item())
            train_offset_losses.append(offset_loss.item())
            train_velocity_losses.append(velocity_loss.item())
            train_activation_losses.append(activation_loss.item())
            
            # Free up memory
            del audio_features, target_onsets, target_offsets, target_velocities
            del loss, onset_loss, offset_loss, velocity_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss = np.mean(train_losses)
        train_onset_loss = np.mean(train_onset_losses)
        train_offset_loss = np.mean(train_offset_losses)
        train_velocity_loss = np.mean(train_velocity_losses)
        train_activation_loss = np.mean(train_activation_losses)
        
        # Monitor prediction statistics on last training batch
        onset_stats = calculate_prediction_stats(onset_logits, threshold=0.5)
        offset_stats = calculate_prediction_stats(offset_logits, threshold=0.5)
        print(f"Training Onset Stats: min={onset_stats['min']:.4f}, max={onset_stats['max']:.4f}, mean={onset_stats['mean']:.4f}, % positive={onset_stats['percent_positive']:.2f}%")
        print(f"Training Offset Stats: min={offset_stats['min']:.4f}, max={offset_stats['max']:.4f}, mean={offset_stats['mean']:.4f}, % positive={offset_stats['percent_positive']:.2f}%")
        
        # Validation
        model.eval()
        val_losses = []
        val_onset_losses = []
        val_offset_losses = []
        val_velocity_losses = []
        val_activation_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device and ensure it's the right type
                audio_features, target_onsets, target_offsets, target_velocities = [x.to(device) for x in batch]
                
                # Forward pass
                onset_logits, offset_logits, velocity_logits = model(audio_features)
                
                # Calculate losses
                onset_loss = onset_loss_fn(onset_logits, target_onsets)
                offset_loss = offset_loss_fn(offset_logits, target_offsets)
                
                # For velocity, apply sigmoid to get probabilities, then calculate loss only for active notes
                velocity_probs = torch.sigmoid(velocity_logits)
                velocity_loss = velocity_loss_fn(velocity_probs * target_onsets, target_velocities * target_onsets)
                
                # Add activation target loss - encourages predictions to be more balanced
                # Use logit space for this calculation
                min_activation_target = args.min_activation_target
                activation_loss_onset = torch.nn.functional.mse_loss(
                    torch.sigmoid(onset_logits).mean(), 
                    torch.tensor(min_activation_target, device=device)
                )
                activation_loss_offset = torch.nn.functional.mse_loss(
                    torch.sigmoid(offset_logits).mean(), 
                    torch.tensor(min_activation_target, device=device)
                )
                activation_loss = activation_loss_onset + activation_loss_offset
                
                # Combined loss
                loss = args.onset_weight * onset_loss + args.offset_weight * offset_loss + args.velocity_weight * velocity_loss
                
                # Add activation loss with its weight
                if args.activation_loss_weight > 0:
                    loss = loss + args.activation_loss_weight * activation_loss
                
                # Record losses
                val_losses.append(loss.item())
                val_onset_losses.append(onset_loss.item())
                val_offset_losses.append(offset_loss.item())
                val_velocity_losses.append(velocity_loss.item())
                val_activation_losses.append(activation_loss.item())
                
                # Free up memory
                del audio_features, target_onsets, target_offsets, target_velocities
                del loss, onset_loss, offset_loss, velocity_loss
                
            # Clean up CUDA memory after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        val_loss = np.mean(val_losses)
        val_onset_loss = np.mean(val_onset_losses)
        val_offset_loss = np.mean(val_offset_losses)
        val_velocity_loss = np.mean(val_velocity_losses)
        val_activation_loss = np.mean(val_activation_losses)
        
        # Monitor prediction statistics on last validation batch
        val_onset_stats = calculate_prediction_stats(onset_logits, threshold=0.5)
        val_offset_stats = calculate_prediction_stats(offset_logits, threshold=0.5)
        print(f"Validation Onset Stats: min={val_onset_stats['min']:.4f}, max={val_onset_stats['max']:.4f}, mean={val_onset_stats['mean']:.4f}, % positive={val_onset_stats['percent_positive']:.2f}%")
        print(f"Validation Offset Stats: min={val_offset_stats['min']:.4f}, max={val_offset_stats['max']:.4f}, mean={val_offset_stats['mean']:.4f}, % positive={val_offset_stats['percent_positive']:.2f}%")
        
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
        print(f"Train Loss: {train_loss:.4f} (Onset: {train_onset_loss:.4f}, Offset: {train_offset_loss:.4f}, Velocity: {train_velocity_loss:.4f}, Activation: {train_activation_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Onset: {val_onset_loss:.4f}, Offset: {val_offset_loss:.4f}, Velocity: {val_velocity_loss:.4f}, Activation: {val_activation_loss:.4f})")
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
    
    #save history
    history_path = model_dir / "training_history.pt"
    torch.save(history, history_path)
    
    print(f"\nTraining completed. Final model saved to {final_model_path}")

#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from piano_transformer import PianoTransformer
from dataset_for_training import PianoTranscriptionDataset
import argparse
from pathlib import Path
import pickle
import numpy as np
import os

def load_or_extract_features(audio_path, feature_path, extractor_fn, extractor_args, fallback_shape=(100, 88)):
    """
    Load features from cache if available, or extract and cache them.

    Args:
        audio_path (Path): Path to the input audio file
        feature_path (Path): Path to the cached .pkl feature file
        extractor_fn (callable): Function to extract features (e.g., process_audio_file)
        extractor_args (dict): Arguments to pass to extractor_fn
        fallback_shape (tuple): Shape of dummy fallback in case of failure

    Returns:
        np.ndarray: Feature matrix
    """
    if feature_path.exists():
        try:
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load cached features for {audio_path.name}: {e}")
            print("Recomputing features...")

    try:
        print(f"Extracting features for {audio_path.name}...")
        features = extractor_fn(audio_path, **extractor_args)

        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features cached to {feature_path}")
        return features
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {audio_path.name}: {e}")
        return np.zeros(fallback_shape, dtype=np.float32)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, device):
    ptModel = PianoTransformer().to(device)
    if args.checkpoint:
        ptModel.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model checkpoint from {args.checkpoint}")
    return ptModel

def get_dataloader(args):
    # Use audio and MIDI directories directly
    audio_dir = Path(args.data_dir) / 'train' / 'audio'
    midi_dir = Path(args.data_dir) / 'train' / 'midi'
    features_dir = Path(args.data_dir) / 'features' / 'train'
    
    # Check if directories exist
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")
    
    # Check for audio files
    audio_files = list(audio_dir.glob('*.wav'))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    print(f"Found {len(audio_files)} audio files in training directory")
    
    dataset = PianoTranscriptionDataset(
        audio_dir=audio_dir,
        midi_dir=midi_dir,
        features_dir=features_dir,
        segment_length=args.segment_length,
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check that your audio and MIDI files match correctly.")
    
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (features, targets) in enumerate(loader):
        features, targets = features.to(device), targets.to(device)
        #Gradient accumulation in PyTorch so we need to zero the gradients
        optimizer.zero_grad()
        note_presence = model(features)
        loss = criterion(note_presence, targets)  # Compute loss
        loss.backward()                           # Compute gradients (backpropagation)
        optimizer.step()                          # Update parameters using gradients
        total_loss += loss.item()
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    return total_loss / len(loader)

def train(args):
    device = get_device()
    print(f"Using device: {device}")
    model = get_model(args, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    loader = get_dataloader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), args.model_path)
    print(f"Final model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--model-path', type=str, default='models\piano_transformer\model.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='models\checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--segment-length', type=int, default=1000)
    args = parser.parse_args()
    train(args)

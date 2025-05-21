#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from piano_transformer import PianoTransformer, PianoTranscriptionDataset
import argparse
from pathlib import Path

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(device):
    return PianoTransformer().to(device)

def get_dataloader(args):
    dataset = PianoTranscriptionDataset(
        features_dir=Path(args.data_dir) / 'features' / 'train',
        midi_dir=Path(args.data_dir) / 'train' / 'midi',
        segment_length=args.segment_length
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        #Gradient accumulation in PyTorch so we need to zero the gradients
        optimizer.zero_grad()
        note_presence, _ = model(features)
        loss = criterion(note_presence, targets)  # Compute loss
        loss.backward()                           # Compute gradients (backpropagation)
        optimizer.step()                          # Update parameters using gradients
        total_loss += loss.item()
    return total_loss / len(loader)

def train(args):
    device = get_device()
    model = get_model(device)
    loader = get_dataloader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), args.model_path)
    print(f"Final model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--model-path', type=str, default='model.pt')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--segment-length', type=int, default=1000)
    args = parser.parse_args()
    train(args)

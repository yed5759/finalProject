#!/usr/bin/env python3

import os
import argparse
import torch
import librosa
import soundfile as sf
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from piano_transformer import PianoTransformer, process_audio_file, notes_to_midi

def transcribe_audio(args):
    # Set device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model parameters (from checkpoint or args)
    if model_path.suffix == '.pt':
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'args' in checkpoint:
                model_args = argparse.Namespace(**checkpoint['args'])
                print("Loaded model parameters from checkpoint")
                n_cqt_bins = model_args.n_cqt_bins
                hidden_dim = model_args.hidden_dim
                num_layers = model_args.num_layers
                num_heads = model_args.num_heads
                dropout = model_args.dropout
            else:
                # Just a state dict, use command line args
                n_cqt_bins = args.n_cqt_bins
                hidden_dim = args.hidden_dim
                num_layers = args.num_layers
                num_heads = args.num_heads
                dropout = args.dropout
        except Exception as e:
            print(f"Error loading checkpoint arguments: {e}")
            n_cqt_bins = args.n_cqt_bins
            hidden_dim = args.hidden_dim
            num_layers = args.num_layers
            num_heads = args.num_heads
            dropout = args.dropout
    else:
        # Use command line arguments
        n_cqt_bins = args.n_cqt_bins
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        num_heads = args.num_heads
        dropout = args.dropout
    
    # Create model
    model = PianoTransformer(
        n_cqt_bins=n_cqt_bins,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # Load model weights
    if model_path.suffix == '.pt':
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    print("Model loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input audio
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Processing audio file: {audio_path}")
    
    # Process audio to extract features
    audio_features = process_audio_file(
        audio_path,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_cqt_bins=n_cqt_bins
    )
    
    # Convert to tensor and add batch dimension
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
    
    # Generate piano transcription
    with torch.no_grad():
        pred_onsets, pred_offsets, pred_velocities = model(audio_features)
    
    # Convert logits to probabilities and numpy arrays
    pred_onsets = torch.sigmoid(pred_onsets[0]).cpu().numpy()
    pred_offsets = torch.sigmoid(pred_offsets[0]).cpu().numpy()
    pred_velocities = pred_velocities[0].cpu().numpy()
    
    # Apply thresholds
    pred_onsets_binary = pred_onsets > args.onset_threshold
    pred_offsets_binary = pred_offsets > args.offset_threshold
    
    # Save piano roll visualizations
    if args.save_piano_roll:
        plt.figure(figsize=(12, 8))
        
        # Plot onsets
        plt.subplot(3, 1, 1)
        plt.imshow(pred_onsets.T, aspect='auto', origin='lower', cmap='Blues')
        plt.colorbar(label='Onset Probability')
        plt.title("Predicted Onsets")
        plt.ylabel("MIDI Note")
        
        # Plot offsets
        plt.subplot(3, 1, 2)
        plt.imshow(pred_offsets.T, aspect='auto', origin='lower', cmap='Reds')
        plt.colorbar(label='Offset Probability')
        plt.title("Predicted Offsets")
        plt.ylabel("MIDI Note")
        
        # Plot velocities
        plt.subplot(3, 1, 3)
        plt.imshow(pred_velocities.T, aspect='auto', origin='lower', cmap='Greens')
        plt.colorbar(label='Velocity')
        plt.title("Predicted Velocities")
        plt.ylabel("MIDI Note")
        plt.xlabel("Time (frames)")
        
        plt.tight_layout()
        piano_roll_path = output_dir / f"{audio_path.stem}_piano_roll.png"
        plt.savefig(piano_roll_path)
        plt.close()
        print(f"Saved piano roll visualization to {piano_roll_path}")
    
    # Convert predictions to MIDI
    midi_obj = notes_to_midi(
        pred_onsets_binary,
        pred_offsets_binary,
        pred_velocities,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate
    )
    
    # Save MIDI
    midi_path = output_dir / f"{audio_path.stem}_transcribed.mid"
    midi_obj.write(str(midi_path))
    print(f"Saved transcribed MIDI to {midi_path}")
    
    # Get MIDI stats
    notes = sum(len(instr.notes) for instr in midi_obj.instruments)
    duration = midi_obj.get_end_time()
    print(f"Transcription stats: {notes} notes, {duration:.2f} seconds")
    
    return {
        'midi_path': midi_path,
        'piano_roll_path': piano_roll_path if args.save_piano_roll else None,
        'notes': notes,
        'duration': duration
    }

def main():
    parser = argparse.ArgumentParser(description='Piano Transcription System')
    
    # Audio and model arguments
    parser.add_argument('--audio-file', type=str, required=True,
                        help='Path to input audio file (WAV)')
    parser.add_argument('--model-path', type=str, default='models/piano_transformer/best_model.pt',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    
    # Audio parameters
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length for feature extraction')
    
    # Model parameters (used if not found in checkpoint)
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
    
    # Inference parameters
    parser.add_argument('--onset-threshold', type=float, default=0.5,
                        help='Threshold for onset detection')
    parser.add_argument('--offset-threshold', type=float, default=0.5,
                        help='Threshold for offset detection')
    parser.add_argument('--save-piano-roll', action='store_true',
                        help='Save piano roll visualization')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    transcribe_audio(args)

if __name__ == "__main__":
    main() 
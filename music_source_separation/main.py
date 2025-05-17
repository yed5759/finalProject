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

# Detect and return available device (GPU or CPU)
def get_device(use_cpu):
    if torch.cuda.is_available() and not use_cpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    print("Using CPU")
    return torch.device('cpu')

# Load model checkpoint (.pt file)
def load_checkpoint(model_path, device):
    if model_path.suffix == '.pt':
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print("Checkpoint loaded")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return None

# Extract model hyperparameters from checkpoint or fallback to CLI args
def extract_model_params(checkpoint, args):
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
        print("Loaded model parameters from checkpoint")
        return model_args.n_cqt_bins, model_args.hidden_dim, model_args.num_layers, model_args.num_heads, model_args.dropout

    return args.n_cqt_bins, args.hidden_dim, args.num_layers, args.num_heads, args.dropout

# Load model weights into initialized model
def load_model_weights(model, model_path, checkpoint, device):
    if (
        model_path.suffix == '.pt' and
        isinstance(checkpoint, dict) and
        'model_state_dict' in checkpoint
    ):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

# Create model with parameters from checkpoint or CLI arguments
def create_model(model_path, checkpoint, device, args):
    n_cqt_bins, hidden_dim, num_layers, num_heads, dropout = extract_model_params(checkpoint, args)
    
    model = PianoTransformer(
        n_cqt_bins=n_cqt_bins,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # Load model weights
    load_model_weights(model, model_path, checkpoint, device)
    
    model.eval()
    print("Model loaded successfully")
    return model, n_cqt_bins

# Create output directory if not exists
def prepare_output_directory(output_dir_path):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Extract audio features (e.g., CQT) for model input
def process_audio(audio_path, sample_rate, hop_length, n_cqt_bins):
    print(f"Processing audio file: {audio_path}")
    audio_features = process_audio_file(
        audio_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_cqt_bins=n_cqt_bins
    )
    return audio_features

# Load audio from path and return (Path, features)
def load_and_process_audio(audio_file_path, sample_rate, hop_length, n_cqt_bins):
    # Process input audio
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Process audio to extract features
    audio_features = process_audio(audio_path, sample_rate, hop_length, n_cqt_bins)
    return audio_path, audio_features

# Load and return model + CQT bin count
def get_model(args, device):
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model parameters (from checkpoint or args) and create model
    checkpoint = load_checkpoint(model_path, device)
    model, n_cqt_bins = create_model(model_path, checkpoint, device, args)
    return model, n_cqt_bins

# Run model inference on audio features
def run_inference(model, audio_features, device):
    # Convert to tensor and add batch dimension
    audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
    # Generate piano transcription
    with torch.no_grad():
        pred_onsets, pred_offsets, pred_velocities = model(audio_tensor)
    return pred_onsets[0], pred_offsets[0], pred_velocities[0]

# Process raw model outputs into probabilities and apply thresholds
def process_predictions(pred_onsets, pred_offsets, pred_velocities, onset_threshold, offset_threshold):
    # Convert logits to probabilities and numpy arrays
    pred_onsets = torch.sigmoid(pred_onsets).cpu().numpy()
    pred_offsets = torch.sigmoid(pred_offsets).cpu().numpy()
    pred_velocities = pred_velocities.cpu().numpy()
    
    # Apply thresholds
    pred_onsets_binary = pred_onsets > onset_threshold
    pred_offsets_binary = pred_offsets > offset_threshold
    
    return pred_onsets, pred_offsets, pred_velocities, pred_onsets_binary, pred_offsets_binary

# Visualize onsets, offsets, and velocities as piano roll
def save_piano_roll_figure(output_dir, audio_path, pred_onsets, pred_offsets, pred_velocities):
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
    return piano_roll_path

# Conditionally save piano roll visualization
def maybe_save_piano_roll(output_dir, audio_path, pred_onsets, pred_offsets, pred_velocities, save_flag):
    if not save_flag:
        return None
    # Save piano roll visualizations
    return save_piano_roll_figure(output_dir, audio_path, pred_onsets, pred_offsets, pred_velocities)

# Convert predictions to MIDI, save file, and compute stats
def save_midi_and_get_stats(output_dir, audio_path, pred_onsets_binary, pred_offsets_binary, pred_velocities, hop_length, sample_rate):
    # Convert predictions to MIDI
    midi_obj = notes_to_midi(
        pred_onsets_binary,
        pred_offsets_binary,
        pred_velocities,
        hop_length=hop_length,
        sample_rate=sample_rate
    )
    # Save MIDI
    midi_path = output_dir / f"{audio_path.stem}_transcribed.mid"
    midi_obj.write(str(midi_path))
    print(f"Saved transcribed MIDI to {midi_path}")
    
    # Get MIDI stats
    notes = sum(len(instr.notes) for instr in midi_obj.instruments)
    duration = midi_obj.get_end_time()
    print(f"Transcription stats: {notes} notes, {duration:.2f} seconds")
        
    return midi_path, notes, duration

# Run full inference pipeline and generate outputs (MIDI, piano roll, stats)
def generate_output(model, audio_features, device, args, output_dir, audio_path, n_cqt_bins):
    pred_onsets, pred_offsets, pred_velocities = run_inference(model, audio_features, device)
    
    pred_onsets, pred_offsets, pred_velocities, pred_onsets_binary, pred_offsets_binary = process_predictions(
        pred_onsets, pred_offsets, pred_velocities, args.onset_threshold, args.offset_threshold
    )

    piano_roll_path = maybe_save_piano_roll(output_dir, audio_path, pred_onsets, pred_offsets, pred_velocities, args.save_piano_roll)      
    
    midi_path, notes, duration = save_midi_and_get_stats(
        output_dir,
        audio_path,
        pred_onsets_binary,
        pred_offsets_binary,
        pred_velocities,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate
    )

    return {
        'midi_path': midi_path,
        'piano_roll_path': piano_roll_path,
        'notes': notes,
        'duration': duration
    }

# Full transcription pipeline: device → model → features → inference → output
def transcribe_audio(args):
    # Set device
    device = get_device(args.cpu)    

    # Load the trained model and extract the CQT bin count
    model, n_cqt_bins = get_model(args, device)

    # Create output directory if it doesn't exist
    output_dir = prepare_output_directory(args.output_dir)
    
    # Load the input audio and extract its features
    audio_path, audio_features = load_and_process_audio(
        args.audio_file,
        args.sample_rate,
        args.hop_length,
        n_cqt_bins
    )
    
    # Run inference and generate output files (MIDI, piano roll, stats)
    result = generate_output(
        model,
        audio_features,
        device,
        args,
        output_dir,
        audio_path,
        n_cqt_bins
    )
    return result

# Parse command line arguments and run transcription
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
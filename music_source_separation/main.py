#!/usr/bin/env python3

import os
import argparse
import torch
import librosa
import soundfile as sf
from dataset_for_training import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from piano_transformer import PianoTransformer
from audio_features import process_audio_file
# from midi_utils import notes_to_midi



# Parse command line arguments and run transcription
def main():
    parser = argparse.ArgumentParser(description='Piano Transcription System')
    
    # Audio and model arguments
    parser.add_argument('--audio-file', type=str, required=True,
                        help='Path to input audio file (WAV)')
    parser.add_argument('--model-path', type=str, default='None',
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
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarizing model output')
    parser.add_argument('--save-piano-roll', action='store_true',
                        help='Save piano roll visualization')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    transcribe_audio(args)

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

# Detect and return available device (GPU or CPU)
def get_device(use_cpu):
    if torch.cuda.is_available() and not use_cpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    print("Using CPU")
    return torch.device('cpu')

# Load and return model + CQT bin count
def get_model(args, device):
    # Load model
    if args.model_path is None:
        model_dir = Path('models/piano_transformer')
        # Determine model parameters (from checkpoint or args) and create model
        checkpoints = sorted(model_dir.glob("checkpoint_epoch_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = checkpoints[0] if checkpoints else None
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")        
    return create_model(model_path, device, args)

# Create model with parameters from checkpoint or CLI arguments
def create_model(model_path, device, args):
    checkpoint = None
    # if there is a checkpoint we use then load it
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
    
    n_cqt_bins, hidden_dim, num_layers, num_heads, dropout = extract_model_params(checkpoint, args)
    if checkpoint is None:
        print("No checkpoint loaded — creating model from CLI args.")
    
    model = PianoTransformer(
        n_cqt_bins=n_cqt_bins,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # Load model weights
    if checkpoint is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint.")
    
    model.eval()
    print("Model loaded successfully")
    return model, n_cqt_bins

# Extract model hyperparameters from checkpoint or fallback to CLI args
def extract_model_params(checkpoint, args):
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
        print("Loaded model parameters from checkpoint")
        return model_args.n_cqt_bins, model_args.hidden_dim, model_args.num_layers, model_args.num_heads, model_args.dropout
        
    print("Checkpoint missing model parameters. Falling back to CLI args.")
    return args.n_cqt_bins, args.hidden_dim, args.num_layers, args.num_heads, args.dropout

# Create output directory if not exists
def prepare_output_directory(output_dir_path):
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Load audio from path and return (Path, features)
def load_and_process_audio(audio_file_path, sample_rate, hop_length, n_cqt_bins):
    """
    Load features from features/ if available, otherwise process audio and save features.
    """
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine features directory and feature file path
    features_dir = audio_path.parent / 'features'
    features_dir.mkdir(exist_ok=True)
    feature_file = features_dir / f"{audio_path.stem}_features.pkl"

    # Try to load precomputed features
    if feature_file.exists():
        print(f"Loading precomputed features from {feature_file}")
        with open(feature_file, 'rb') as f:
            audio_features = pickle.load(f)
    else:
        print(f"No precomputed features found. Processing audio and saving to {feature_file}")
        audio_features = process_audio_file(
            audio_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_cqt_bins=n_cqt_bins
        )
        with open(feature_file, 'wb') as f:
            pickle.dump(audio_features, f)
    return audio_path, audio_features

# Run full inference pipeline and generate outputs (MIDI, piano roll, stats)
def generate_output(model, audio_features, device, args, output_dir, audio_path, n_cqt_bins):
    frame_probs = run_inference(model, audio_features, device)
    
    frame_probs, frame_binary = process_predictions(frame_probs, args.threshold)

    piano_roll_path = save_piano_roll_figure(output_dir, audio_path, frame_probs, args.save_piano_roll)      
    
    midi_path, notes, duration = save_midi_and_get_stats(
        output_dir,
        audio_path,
        frame_binary,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate
    )

    return {
        'midi_path': midi_path,
        'piano_roll_path': piano_roll_path,
        'notes': notes,
        'duration': duration
    }

# Run model inference on audio features
def run_inference(model, audio_features, device):
    # Convert to tensor and add batch dimension
    audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
    # Generate piano transcription
    with torch.no_grad():
       frame_probs = model(audio_tensor)
    return frame_probs[0]

# Process raw model outputs into probabilities and apply thresholds
def process_predictions(frame_probs, threshold):
    # Convert logits to probabilities and numpy arrays
    if isinstance(frame_probs, torch.Tensor):
        frame_probs = torch.sigmoid(frame_probs).cpu().numpy()

    frame_binary = frame_probs > threshold
    return frame_probs, frame_binary

# Visualize piano roll
def save_piano_roll_figure(output_dir, audio_path, frame_probs, save_flag):
    if not save_flag:
        return None

    plt.figure(figsize=(12, 4))
    
    # Plot frame-based piano roll
    plt.imshow(frame_probs.T, aspect='auto', origin='lower', cmap='Blues')
    plt.colorbar(label='Note Activation Probability')
    plt.title("Predicted Frame-Based Piano Roll")
    plt.ylabel("MIDI Note")
    plt.xlabel("Time (frames)")
    
    plt.tight_layout()
    piano_roll_path = output_dir / f"{audio_path.stem}_piano_roll.png"
    plt.savefig(piano_roll_path)
    plt.close()
    print(f"Saved piano roll visualization to {piano_roll_path}")
    return piano_roll_path

# Convert predictions to MIDI, save file, and compute stats
def save_midi_and_get_stats(output_dir, audio_path, frame_binary, hop_length, sample_rate):
    # Convert predictions to MIDI
    midi_obj = notes_to_midi(
        frame_binary,
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

if __name__ == "__main__":
    main() 

#!/usr/bin/env python3

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error

from piano_transformer import PianoTransformer, process_audio_file, midi_to_piano_roll

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model parameters (either from checkpoint or user arguments)
    if model_path.suffix == '.pt':
        # Try to load a checkpoint that includes arguments
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
    
    # Get test files
    test_audio_dir = Path(args.data_dir) / 'test' / 'audio'
    test_midi_dir = Path(args.data_dir) / 'test' / 'midi'
    
    if not test_audio_dir.exists() or not test_midi_dir.exists():
        raise FileNotFoundError(f"Test data not found in {args.data_dir}/test")
    
    audio_files = sorted(list(test_audio_dir.glob('*.wav')))
    midi_files = []
    
    for audio_file in audio_files:
        midi_file = test_midi_dir / f"{audio_file.stem}.mid"
        if midi_file.exists():
            midi_files.append(midi_file)
        else:
            print(f"Warning: No matching MIDI file for {audio_file}")
            audio_files.remove(audio_file)
    
    print(f"Found {len(audio_files)} test files with matching MIDI")
    
    # Evaluation metrics
    metrics = {
        'onset_precision': [],
        'onset_recall': [],
        'onset_f1': [],
        'offset_precision': [],
        'offset_recall': [],
        'offset_f1': [],
        'velocity_rmse': []
    }
    
    # Process each test file
    for i, (audio_file, midi_file) in enumerate(zip(audio_files, midi_files)):
        print(f"\nProcessing file {i+1}/{len(audio_files)}: {audio_file.name}")
        
        # Process audio
        audio_features = process_audio_file(
            audio_file, 
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_cqt_bins=n_cqt_bins
        )
        
        # Convert to tensor and add batch dimension
        audio_features = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
        
        # Get ground truth piano roll
        ground_truth_midi = pretty_midi.PrettyMIDI(str(midi_file))
        ground_truth_onsets, ground_truth_offsets, ground_truth_velocities = midi_to_piano_roll(
            ground_truth_midi,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            roll_length=audio_features.shape[1]
        )
        
        # Convert to tensors
        ground_truth_onsets = torch.FloatTensor(ground_truth_onsets)
        ground_truth_offsets = torch.FloatTensor(ground_truth_offsets)
        ground_truth_velocities = torch.FloatTensor(ground_truth_velocities)
        
        # Make predictions
        with torch.no_grad():
            pred_onsets, pred_offsets, pred_velocities = model(audio_features)
        
        # Convert to numpy
        pred_onsets = torch.sigmoid(pred_onsets[0]).cpu().numpy()
        pred_offsets = torch.sigmoid(pred_offsets[0]).cpu().numpy()
        pred_velocities = pred_velocities[0].cpu().numpy()
        
        # Apply threshold to onsets/offsets
        pred_onsets_binary = pred_onsets > args.onset_threshold
        pred_offsets_binary = pred_offsets > args.offset_threshold
        
        # Calculate metrics
        # Flatten for sklearn metrics
        gt_onsets_flat = ground_truth_onsets.numpy().flatten()
        pred_onsets_flat = pred_onsets_binary.flatten()
        
        gt_offsets_flat = ground_truth_offsets.numpy().flatten()
        pred_offsets_flat = pred_offsets_binary.flatten()
        
        # Only evaluate velocity on notes that exist (where onset=1)
        mask = gt_onsets_flat > 0
        gt_velocities_masked = ground_truth_velocities.numpy().flatten()[mask] 
        pred_velocities_masked = pred_velocities.flatten()[mask]
        
        # Calculate precision, recall, F1
        onset_precision, onset_recall, onset_f1, _ = precision_recall_fscore_support(
            gt_onsets_flat, pred_onsets_flat, average='binary', zero_division=0
        )
        
        offset_precision, offset_recall, offset_f1, _ = precision_recall_fscore_support(
            gt_offsets_flat, pred_offsets_flat, average='binary', zero_division=0
        )
        
        # Calculate velocity RMSE
        if len(gt_velocities_masked) > 0:
            velocity_rmse = np.sqrt(mean_squared_error(gt_velocities_masked, pred_velocities_masked))
        else:
            velocity_rmse = float('nan')
        
        # Store metrics
        metrics['onset_precision'].append(onset_precision)
        metrics['onset_recall'].append(onset_recall)
        metrics['onset_f1'].append(onset_f1)
        metrics['offset_precision'].append(offset_precision)
        metrics['offset_recall'].append(offset_recall)
        metrics['offset_f1'].append(offset_f1)
        metrics['velocity_rmse'].append(velocity_rmse)
        
        # Print metrics for this file
        print(f"Onset - Precision: {onset_precision:.4f}, Recall: {onset_recall:.4f}, F1: {onset_f1:.4f}")
        print(f"Offset - Precision: {offset_precision:.4f}, Recall: {offset_recall:.4f}, F1: {offset_f1:.4f}")
        print(f"Velocity RMSE: {velocity_rmse:.4f}")
        
        # Generate and save piano roll visualizations if requested
        if args.visualize:
            plt.figure(figsize=(15, 10))
            
            # Plot ground truth onsets
            plt.subplot(3, 2, 1)
            plt.imshow(ground_truth_onsets.numpy().T, aspect='auto', origin='lower', cmap='Blues')
            plt.title("Ground Truth Onsets")
            plt.ylabel("MIDI Note")
            
            # Plot predicted onsets
            plt.subplot(3, 2, 2)
            plt.imshow(pred_onsets.T, aspect='auto', origin='lower', cmap='Blues')
            plt.title("Predicted Onsets")
            
            # Plot ground truth offsets
            plt.subplot(3, 2, 3)
            plt.imshow(ground_truth_offsets.numpy().T, aspect='auto', origin='lower', cmap='Reds')
            plt.title("Ground Truth Offsets")
            plt.ylabel("MIDI Note")
            
            # Plot predicted offsets
            plt.subplot(3, 2, 4)
            plt.imshow(pred_offsets.T, aspect='auto', origin='lower', cmap='Reds')
            plt.title("Predicted Offsets")
            
            # Plot ground truth velocities
            plt.subplot(3, 2, 5)
            plt.imshow(ground_truth_velocities.numpy().T, aspect='auto', origin='lower', cmap='Greens')
            plt.title("Ground Truth Velocities")
            plt.ylabel("MIDI Note")
            plt.xlabel("Time (frames)")
            
            # Plot predicted velocities
            plt.subplot(3, 2, 6)
            plt.imshow(pred_velocities.T, aspect='auto', origin='lower', cmap='Greens')
            plt.title("Predicted Velocities")
            plt.xlabel("Time (frames)")
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{audio_file.stem}_piano_roll.png")
            plt.close()
            
            # Generate MIDI from predictions
            if args.generate_midi:
                predicted_midi = notes_to_midi(
                    pred_onsets_binary,
                    pred_offsets_binary,
                    pred_velocities,
                    hop_length=args.hop_length,
                    sample_rate=args.sample_rate
                )
                
                predicted_midi.write(output_dir / f"{audio_file.stem}_predicted.mid")
    
    # Calculate average metrics
    avg_metrics = {k: np.nanmean(v) for k, v in metrics.items()}
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Onset - Precision: {avg_metrics['onset_precision']:.4f}, Recall: {avg_metrics['onset_recall']:.4f}, F1: {avg_metrics['onset_f1']:.4f}")
    print(f"Offset - Precision: {avg_metrics['offset_precision']:.4f}, Recall: {avg_metrics['offset_recall']:.4f}, F1: {avg_metrics['offset_f1']:.4f}")
    print(f"Velocity RMSE: {avg_metrics['velocity_rmse']:.4f}")
    
    # Save metrics to file
    with open(output_dir / "evaluation_metrics.txt", "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Onset - Precision: {avg_metrics['onset_precision']:.4f}, Recall: {avg_metrics['onset_recall']:.4f}, F1: {avg_metrics['onset_f1']:.4f}\n")
        f.write(f"Offset - Precision: {avg_metrics['offset_precision']:.4f}, Recall: {avg_metrics['offset_recall']:.4f}, F1: {avg_metrics['offset_f1']:.4f}\n")
        f.write(f"Velocity RMSE: {avg_metrics['velocity_rmse']:.4f}\n")
        
        f.write("\nDetailed Metrics:\n")
        for i, (audio_file, midi_file) in enumerate(zip(audio_files, midi_files)):
            f.write(f"\nFile {i+1}: {audio_file.name}\n")
            f.write(f"Onset - Precision: {metrics['onset_precision'][i]:.4f}, Recall: {metrics['onset_recall'][i]:.4f}, F1: {metrics['onset_f1'][i]:.4f}\n")
            f.write(f"Offset - Precision: {metrics['offset_precision'][i]:.4f}, Recall: {metrics['offset_recall'][i]:.4f}, F1: {metrics['offset_f1'][i]:.4f}\n")
            f.write(f"Velocity RMSE: {metrics['velocity_rmse'][i]:.4f}\n")
    
    print(f"Evaluation complete. Results saved to {output_dir}")

def notes_to_midi(onsets, offsets, velocities, hop_length=512, sample_rate=16000):
    """Convert piano roll matrices to MIDI file"""
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Frame timing
    frame_time = hop_length / sample_rate
    
    # Iterate through each note (88 piano keys, starting from A0=21)
    for note_idx in range(88):
        midi_note = note_idx + 21  # A0 = 21
        
        # Find onset frames
        onset_frames = np.where(onsets[:, note_idx])[0]
        
        # For each detected onset
        for onset_frame in onset_frames:
            # Find the next offset after this onset
            offset_frames = np.where(offsets[onset_frame:, note_idx])[0]
            
            if len(offset_frames) > 0:
                # Offset is relative to onset_frame, so add it back
                offset_frame = offset_frames[0] + onset_frame
            else:
                # If no offset found, set to end of song
                offset_frame = len(onsets) - 1
            
            # Convert frames to time
            start_time = onset_frame * frame_time
            end_time = offset_frame * frame_time
            
            # Ensure note has minimum duration
            if end_time <= start_time:
                end_time = start_time + 0.1  # Minimum 100ms note
            
            # Get velocity (scale to 0-127 for MIDI)
            velocity = int(min(max(velocities[onset_frame, note_idx] * 127, 1), 127))
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=midi_note,
                start=start_time,
                end=end_time
            )
            
            # Add note to instrument
            piano.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(piano)
    return midi

def main():
    parser = argparse.ArgumentParser(description='Evaluate piano transcription model')
    
    # Data and model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default='output/evaluation',
                        help='Directory to save evaluation results')
    
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
    
    # Audio parameters
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length for feature extraction')
    
    # Evaluation parameters
    parser.add_argument('--onset-threshold', type=float, default=0.5,
                        help='Threshold for onset detection')
    parser.add_argument('--offset-threshold', type=float, default=0.5,
                        help='Threshold for offset detection')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate piano roll visualizations')
    parser.add_argument('--generate-midi', action='store_true',
                        help='Generate MIDI files from predictions')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main() 
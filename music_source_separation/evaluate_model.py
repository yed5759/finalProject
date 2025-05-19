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
import shutil
import gc

from piano_transformer import PianoTransformer, process_audio_file, midi_to_piano_roll

def load_model(args, device):
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint args or fallback to user args
    n_cqt_bins = args.n_cqt_bins
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    dropout = args.dropout
    
    # Determine model parameters (either from checkpoint or user arguments)
    checkpoint = None
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
        except Exception as e:
            print(f"Error loading checkpoint arguments: {e}")
    
    # Create model
    model = PianoTransformer(
        n_cqt_bins=n_cqt_bins,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # Load model weights
    if checkpoint and isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    print("Model loaded successfully")
    return model, n_cqt_bins

def load_test_files(data_dir):
    test_audio_dir = Path(data_dir) / 'test' / 'audio'
    test_midi_dir = Path(data_dir) / 'test' / 'midi'
    if not test_audio_dir.exists() or not test_midi_dir.exists():
        raise FileNotFoundError(f"Test data not found in {data_dir}/test")
    
    audio_files = sorted(list(test_audio_dir.glob('*.wav')))
    midi_files = []
    for audio_file in audio_files[:]:
        midi_file = test_midi_dir / f"{audio_file.stem}.mid"
        if midi_file.exists():
            midi_files.append(midi_file)
        else:
            print(f"Warning: No matching MIDI file for {audio_file}")
            audio_files.remove(audio_file)
    print(f"Found {len(audio_files)} test files with matching MIDI")
    return audio_files, midi_files

def initialize_metrics():
    return {
        'onset_precision': [],
        'onset_recall': [],
        'onset_f1': [],
        'offset_precision': [],
        'offset_recall': [],
        'offset_f1': [],
        'velocity_rmse': []
    }

def predict_chunks(model, features, device, max_seq_len):
    # Initialize prediction arrays
    pred_onsets = np.zeros((len(features), 88))
    pred_offsets = np.zeros((len(features), 88))
    pred_velocities = np.zeros((len(features), 88))

    # Process in chunks with overlap
    chunk_size = max_seq_len
    overlap = min(chunk_size // 4, 1000) # 25% overlap, max 1000 frames


    for start_idx in range(0, len(features), chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, len(features))
        chunk_features = features[start_idx:end_idx]
        chunk_length = len(chunk_features)

        # Convert to tensor and add batch dimension
        chunk_tensor = torch.FloatTensor(chunk_features).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            chunk_onsets, chunk_offsets, chunk_velocities = model(chunk_tensor)

        # Convert to numpy - no need for sigmoid as it's already applied in the model
        chunk_onsets = chunk_onsets[0].cpu().numpy()
        chunk_offsets = chunk_offsets[0].cpu().numpy()
        chunk_velocities = chunk_velocities[0].cpu().numpy()

        if start_idx == 0:
            # First chunk
            pred_onsets[:end_idx] = chunk_onsets
            pred_offsets[:end_idx] = chunk_offsets
            pred_velocities[:end_idx] = chunk_velocities
        else:
            # Handle overlap with linear blending
            blend_start = start_idx
            blend_end = min(start_idx + overlap, len(features))
            blend_length = blend_end - blend_start

            # Create linear weights for blending
            old_weight = np.linspace(1, 0, blend_length).reshape(-1, 1)
            new_weight = np.linspace(0, 1, blend_length).reshape(-1, 1)
            
            # Blend overlap region
            pred_onsets[blend_start:blend_end] = old_weight * pred_onsets[blend_start:blend_end] + new_weight * chunk_onsets[:blend_length]
            pred_offsets[blend_start:blend_end] = old_weight * pred_offsets[blend_start:blend_end] + new_weight * chunk_offsets[:blend_length]
            pred_velocities[blend_start:blend_end] = old_weight * pred_velocities[blend_start:blend_end] + new_weight * chunk_velocities[:blend_length]

            if end_idx > blend_end:
                # Copy non-overlap region
                pred_onsets[blend_end:end_idx] = chunk_onsets[blend_length:chunk_length]
                pred_offsets[blend_end:end_idx] = chunk_offsets[blend_length:chunk_length]
                pred_velocities[blend_end:end_idx] = chunk_velocities[blend_length:chunk_length]

        # Free up GPU memory
        del chunk_tensor, chunk_onsets, chunk_offsets, chunk_velocities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Report progress
        progress = (end_idx / len(features)) * 100
        print(f"  Processed {end_idx}/{len(features)} frames ({progress:.1f}%)")

        # Check if we've reached the end
        if end_idx == len(features):
            break

    return pred_onsets, pred_offsets, pred_velocities

def predict_full(model, features, device):
    # Convert to tensor and add batch dimension
    audio_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make predictions
    with torch.no_grad():
        pred_onsets, pred_offsets, pred_velocities = model(audio_tensor)
    
    # Convert to numpy - no need for sigmoid as it's already applied in the model
    pred_onsets = pred_onsets[0].cpu().numpy()
    pred_offsets = pred_offsets[0].cpu().numpy()
    pred_velocities = pred_velocities[0].cpu().numpy()
    
    # Free up GPU memory
    del audio_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return pred_onsets, pred_offsets, pred_velocities

def calculate_metrics(ground_truth_onsets, ground_truth_offsets, ground_truth_velocities,
                      pred_onsets, pred_offsets, pred_velocities, args):
    # Apply threshold to onsets/offsets
    pred_onsets_binary = pred_onsets > args.onset_threshold
    pred_offsets_binary = pred_offsets > args.offset_threshold

    # Calculate metrics
    # Flatten for sklearn metrics
    gt_onsets_flat = ground_truth_onsets.flatten()
    pred_onsets_flat = pred_onsets_binary.flatten()

    gt_offsets_flat = ground_truth_offsets.flatten()
    pred_offsets_flat = pred_offsets_binary.flatten()

    # Only evaluate velocity on notes that exist (where onset=1)
    mask = gt_onsets_flat > 0
    gt_velocities_masked = ground_truth_velocities.flatten()[mask]
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

    return {
        'onset_precision': onset_precision,
        'onset_recall': onset_recall,
        'onset_f1': onset_f1,
        'offset_precision': offset_precision,
        'offset_recall': offset_recall,
        'offset_f1': offset_f1,
        'velocity_rmse': velocity_rmse
    }

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, n_cqt_bins = load_model(args, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get test files
    audio_files, midi_files = load_test_files(args.data_dir)
    
    # Evaluation metrics
    metrics = initialize_metrics()
    
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
        
        # Get ground truth piano roll
        ground_truth_midi = pretty_midi.PrettyMIDI(str(midi_file))
        ground_truth_onsets, ground_truth_offsets, ground_truth_velocities = midi_to_piano_roll(
            ground_truth_midi,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            roll_length=len(audio_features)
        )
        
        # Handle long sequences with chunking if needed
        if len(audio_features) > args.max_sequence_length:
            print(f"Long sequence detected: {len(audio_features)} frames. Using chunked processing.")
            
            pred_onsets, pred_offsets, pred_velocities = predict_chunks(
                model, audio_features, device, args.max_sequence_length
            )

        else:
            # For shorter sequences, process the entire audio at once
            pred_onsets, pred_offsets, pred_velocities = predict_full(
                model, audio_features, device
            )

        # Add debug output to see prediction statistics
        print(f"Onset prediction stats - min: {pred_onsets.min():.4f}, max: {pred_onsets.max():.4f}, mean: {pred_onsets.mean():.4f}")
        print(f"Offset prediction stats - min: {pred_offsets.min():.4f}, max: {pred_offsets.max():.4f}, mean: {pred_offsets.mean():.4f}")
        print(f"Onset predictions > threshold: {(pred_onsets > args.onset_threshold).sum()} out of {pred_onsets.size}")
        print(f"Offset predictions > threshold: {(pred_offsets > args.offset_threshold).sum()} out of {pred_offsets.size}")
        print(f"Ground truth onset positives: {ground_truth_onsets.sum()} out of {ground_truth_onsets.size}")
        print(f"Ground truth offset positives: {ground_truth_offsets.sum()} out of {ground_truth_offsets.size}")

        # Evaluate performance
        file_metrics = calculate_metrics(
            ground_truth_onsets,
            ground_truth_offsets,
            ground_truth_velocities,
            pred_onsets,
            pred_offsets,
            pred_velocities
        )

        # Store metrics
        metrics['onset_precision'].append(file_metrics['onset_precision'])
        metrics['onset_recall'].append(file_metrics['onset_recall'])
        metrics['onset_f1'].append(file_metrics['onset_f1'])
        metrics['offset_precision'].append(file_metrics['offset_precision'])
        metrics['offset_recall'].append(file_metrics['offset_recall'])
        metrics['offset_f1'].append(file_metrics['offset_f1'])
        metrics['velocity_rmse'].append(file_metrics['velocity_rmse'])
            
        # Print metrics for this file
        print(f"Onset - Precision: {file_metrics['onset_precision']:.4f}, Recall: {file_metrics['onset_recall']:.4f}, F1: {file_metrics['onset_f1']:.4f}")
        print(f"Offset - Precision: {file_metrics['offset_precision']:.4f}, Recall: {file_metrics['offset_recall']:.4f}, F1: {file_metrics['offset_f1']:.4f}")
        print(f"Velocity RMSE: {file_metrics['velocity_rmse']:.4f}")
        
        # Generate MIDI from predictions if requested
        if args.generate_midi:
            predicted_midi = notes_to_midi(
                pred_onsets_binary,
                pred_offsets_binary,
                pred_velocities,
                hop_length=args.hop_length,
                sample_rate=args.sample_rate
            )
            
            midi_path = output_dir / f"{audio_file.stem}_predicted.mid"
            predicted_midi.write(str(midi_path))
            print(f"Generated MIDI saved to {midi_path}")
            
            # Also save ground truth MIDI for comparison
            truth_midi_path = output_dir / f"{audio_file.stem}_ground_truth.mid"
            shutil.copy(midi_file, truth_midi_path)
        
        # Generate and save piano roll visualizations if requested
        if args.visualize:
            # Generate piano roll visualization (simplified to save space)
            plt.figure(figsize=(15, 10))
            
            # Plot predicted vs ground truth onsets (just a sample)
            plt.subplot(2, 1, 1)
            # Limit visualization to a reasonable range (first 1000 frames or less)
            display_len = min(1000, len(audio_features))
            plt.imshow(np.vstack([
                ground_truth_onsets[:display_len, :].T,
                np.zeros((5, display_len)),  # Add a separator
                pred_onsets_binary[:display_len, :].T
            ]), aspect='auto', cmap='Blues')
            plt.title(f"Piano Roll: Ground Truth (top) vs Prediction (bottom) - First {display_len} frames")
            plt.ylabel("MIDI Note")
            
            plt.subplot(2, 1, 2)
            plt.plot(np.sum(ground_truth_onsets[:display_len], axis=1), label='Ground Truth Onsets')
            plt.plot(np.sum(pred_onsets_binary[:display_len], axis=1), label='Predicted Onsets')
            plt.xlabel("Time (frames)")
            plt.ylabel("Number of active notes")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{audio_file.stem}_piano_roll.png")
            plt.close()
            
        # Free up memory
        del pred_onsets, pred_offsets, pred_velocities, pred_onsets_binary, pred_offsets_binary
        del ground_truth_onsets, ground_truth_offsets, ground_truth_velocities
        gc.collect()
    
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
    parser.add_argument('--max-sequence-length', type=int, default=10000,
                        help='Maximum sequence length to process at once (longer sequences will be chunked)')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main() 
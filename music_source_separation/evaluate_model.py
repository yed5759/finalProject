#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import mir_eval
import pretty_midi

from piano_transformer import PianoTranscriptionSystem

def evaluate_transcription(reference_file, estimated_file):
    """
    Evaluate transcription using mir_eval metrics
    
    Args:
        reference_file: Path to reference MIDI file
        estimated_file: Path to estimated MIDI file
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Load reference MIDI
    ref_midi = pretty_midi.PrettyMIDI(str(reference_file))
    ref_intervals, ref_pitches = ref_midi.get_piano_roll_intervals_and_pitches()
    
    # Load estimated MIDI
    est_midi = pretty_midi.PrettyMIDI(str(estimated_file))
    est_intervals, est_pitches = est_midi.get_piano_roll_intervals_and_pitches()
    
    # Convert pitches to midi note numbers if needed
    if ref_pitches.dtype.kind == 'f':
        ref_pitches = np.array(ref_pitches, dtype=np.int)
    if est_pitches.dtype.kind == 'f':
        est_pitches = np.array(est_pitches, dtype=np.int)
    
    # Evaluate
    # If there are no notes, return placeholder metrics
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'error_rate': 1.0
        }
    
    # Calculate metrics
    results = mir_eval.transcription.evaluate(
        ref_intervals, 
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,  # 50ms tolerance
        offset_ratio=0.2  # 20% of the note length
    )
    
    # Extract key metrics
    metrics = {
        'precision': results['Precision'],
        'recall': results['Recall'],
        'f1': results['F-measure'],
        'accuracy': results['Accuracy'],
        'error_rate': 1.0 - results['Accuracy']
    }
    
    return metrics

def evaluate_model(args):
    """Evaluate the piano transcription system on the test set"""
    
    # Initialize transcription system
    system = PianoTranscriptionSystem(
        model_path=args.model_path,
        device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    
    # Get test data paths
    data_dir = Path(args.data_dir)
    test_audio_dir = data_dir / 'test' / 'audio'
    test_midi_dir = data_dir / 'test' / 'midi'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results directory
    results_dir = output_dir / 'transcriptions'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all test files
    test_audio_files = sorted(list(test_audio_dir.glob('*.wav')))
    midi_files = {f.stem: f for f in test_midi_dir.glob('*.mid')}
    
    # Filter to files that have both audio and MIDI
    test_pairs = []
    for audio_file in test_audio_files:
        if audio_file.stem in midi_files:
            test_pairs.append({
                'audio': audio_file,
                'midi': midi_files[audio_file.stem]
            })
    
    # Limit the number of files if specified
    if args.max_files is not None:
        test_pairs = test_pairs[:args.max_files]
    
    print(f"Evaluating on {len(test_pairs)} test files")
    
    # Evaluate each file
    all_metrics = []
    for pair in tqdm(test_pairs):
        # Transcribe audio
        output_file = results_dir / f"{pair['audio'].stem}_transcribed.mid"
        
        try:
            # Transcribe audio to MIDI
            system.transcribe(
                pair['audio'],
                output_file,
                sample_rate=args.sample_rate,
                n_fft=args.fft_size,
                feature_type=args.feature_type
            )
            
            # Evaluate transcription
            metrics = evaluate_transcription(pair['midi'], output_file)
            metrics['file'] = pair['audio'].stem
            all_metrics.append(metrics)
            
            # Print metrics for this file
            print(f"{pair['audio'].stem}: F1={metrics['f1']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"Error processing {pair['audio']}: {e}")
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'error_rate': np.mean([m['error_rate'] for m in all_metrics])
    }
    
    # Save all results
    results = {
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'model_path': args.model_path,
        'feature_type': args.feature_type,
        'sample_rate': args.sample_rate,
        'num_files': len(test_pairs)
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of files: {len(test_pairs)}")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Error Rate: {avg_metrics['error_rate']:.4f}")
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate piano transcription model')
    
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                      help='Path to output directory')
    parser.add_argument('--feature-type', type=str, default='mel', choices=['mel', 'cqt', 'both'],
                      help='Feature type for audio representation')
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050],
                      help='Audio sample rate')
    parser.add_argument('--fft-size', type=int, default=2048,
                      help='FFT window size')
    parser.add_argument('--max-files', type=int, default=None,
                      help='Maximum number of files to evaluate')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage')
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main() 
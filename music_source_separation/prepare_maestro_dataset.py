#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa

def process_audio(input_file, output_file, sample_rate):
    """
    Process audio file to correct sample rate and format
    """
    # Load audio with librosa (handles resampling)
    audio, _ = librosa.load(input_file, sr=sample_rate, mono=True)
    
    # Save as WAV with specified sample rate
    sf.write(output_file, audio, sample_rate, format='WAV')
    
    return len(audio) / sample_rate  # Return duration in seconds

def prepare_maestro(maestro_dir, output_dir, sample_rate=16000):
    """
    Prepare MAESTRO dataset for training
    
    Args:
        maestro_dir: Path to MAESTRO dataset
        output_dir: Path to output directory
        sample_rate: Target sample rate
    """
    maestro_dir = Path(maestro_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_audio_dir = output_dir / "train" / "audio"
    train_midi_dir = output_dir / "train" / "midi"
    val_audio_dir = output_dir / "val" / "audio"
    val_midi_dir = output_dir / "val" / "midi"
    test_audio_dir = output_dir / "test" / "audio"
    test_midi_dir = output_dir / "test" / "midi"
    
    os.makedirs(train_audio_dir, exist_ok=True)
    os.makedirs(train_midi_dir, exist_ok=True)
    os.makedirs(val_audio_dir, exist_ok=True)
    os.makedirs(val_midi_dir, exist_ok=True)
    os.makedirs(test_audio_dir, exist_ok=True)
    os.makedirs(test_midi_dir, exist_ok=True)
    
    # Load MAESTRO metadata
    json_path = maestro_dir / "maestro-v3.0.0.json"
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    
    # Try to load metadata from JSON or CSV
    if json_path.exists():
        with open(json_path, 'r') as f:
            maestro_data = json.load(f)
            items = maestro_data.get('items', [])
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        items = df.to_dict('records')
    else:
        raise FileNotFoundError(f"Could not find metadata file in {maestro_dir}")
    
    # Process files
    stats = {
        'train': {'count': 0, 'duration': 0},
        'validation': {'count': 0, 'duration': 0},
        'test': {'count': 0, 'duration': 0}
    }
    
    print(f"Processing MAESTRO dataset with {len(items)} items...")
    for item in tqdm(items):
        # Get paths and split
        audio_filename = item.get('audio_filename') or item.get('audio')
        midi_filename = item.get('midi_filename') or item.get('midi')
        split = item.get('split')
        
        if not audio_filename or not midi_filename or not split:
            print(f"Skipping item with missing information: {item}")
            continue
        
        # Handle different path formats
        audio_path = maestro_dir / audio_filename
        midi_path = maestro_dir / midi_filename
        
        # If paths don't exist, try alternate formats
        if not audio_path.exists() and '/' in audio_filename:
            # Path might be relative to MAESTRO root
            audio_path = maestro_dir / Path(audio_filename)
        if not midi_path.exists() and '/' in midi_filename:
            midi_path = maestro_dir / Path(midi_filename)
        
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            continue
        if not midi_path.exists():
            print(f"MIDI file not found: {midi_path}")
            continue
        
        # Determine output paths
        if split == 'train':
            output_audio_dir = train_audio_dir
            output_midi_dir = train_midi_dir
        elif split == 'validation':
            output_audio_dir = val_audio_dir
            output_midi_dir = val_midi_dir
        elif split == 'test':
            output_audio_dir = test_audio_dir
            output_midi_dir = test_midi_dir
        else:
            print(f"Unknown split: {split}, skipping")
            continue
        
        # Generate output filenames
        output_basename = f"{Path(audio_filename).stem}"
        output_audio_path = output_audio_dir / f"{output_basename}.wav"
        output_midi_path = output_midi_dir / f"{output_basename}.mid"
        
        # Process audio
        try:
            duration = process_audio(audio_path, output_audio_path, sample_rate)
            # Copy MIDI file
            shutil.copy2(midi_path, output_midi_path)
            
            # Update stats
            stats[split]['count'] += 1
            stats[split]['duration'] += duration
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    # Print statistics
    print("\nDataset Statistics:")
    for split, data in stats.items():
        hours = data['duration'] / 3600
        print(f"{split}: {data['count']} files, {hours:.2f} hours")
    
    # Create a summary file
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump({
            'stats': stats,
            'sample_rate': sample_rate,
            'prepare_date': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"\nMAESTRO dataset prepared successfully in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare MAESTRO dataset for piano transcription')
    parser.add_argument('--maestro-dir', type=str, required=True, 
                      help='Path to MAESTRO dataset')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Path to output directory')
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050],
                      help='Target sample rate')
    
    args = parser.parse_args()
    
    prepare_maestro(args.maestro_dir, args.output_dir, args.sample_rate)

if __name__ == "__main__":
    main() 
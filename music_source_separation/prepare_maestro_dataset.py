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

def prepare_maestro_dataset(args):
    """
    Prepare MAESTRO dataset for training, validation, and testing
    Works with existing dataset structure in dataset/MAESTRO
    """
    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for subdir in ['audio', 'midi']:
            os.makedirs(output_dir / split / subdir, exist_ok=True)
    
    # Load metadata
    metadata_path = maestro_dir / 'maestro-v3.0.0.json'
    if not metadata_path.exists():
        metadata_path = maestro_dir / 'maestro-v3.0.0.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find MAESTRO metadata at {metadata_path}")
        metadata = pd.read_csv(metadata_path)
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata = pd.DataFrame(metadata['items'])
    
    # Group files by split
    train_files = metadata[metadata['split'] == 'train']
    validation_files = metadata[metadata['split'] == 'validation']
    test_files = metadata[metadata['split'] == 'test']
    
    print(f"Found {len(train_files)} training files, {len(validation_files)} validation files, "
          f"and {len(test_files)} test files")
    
    # Process files by split
    total_files = 0
    processed_files = 0
    
    for split, files in zip(['train', 'val', 'test'], [train_files, validation_files, test_files]):
        print(f"Processing {split} files...")
        
        for _, row in tqdm(files.iterrows(), total=len(files)):
            audio_filename = row['audio_filename']
            midi_filename = row['midi_filename']
            
            # Locate files in the existing structure
            # Example: dataset/MAESTRO/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi
            year = str(row.get('year', audio_filename.split('/')[0]))
            
            # Check for different filename formats
            if os.path.basename(audio_filename).endswith('.wav'):
                audio_id = os.path.basename(audio_filename).replace('.wav', '')
            else:
                audio_id = os.path.basename(audio_filename)
            
            # Find the audio file in the dataset structure
            source_audio = None
            source_midi = None
            
            # Try to find the audio file with exact path
            if (maestro_dir / audio_filename).exists():
                source_audio = maestro_dir / audio_filename
            else:
                # Try to find by year and filename
                year_dir = maestro_dir / year
                if year_dir.exists():
                    for file in year_dir.glob('*.wav'):
                        if audio_id in file.stem:
                            source_audio = file
                            break
            
            # Try to find the MIDI file with exact path
            if (maestro_dir / midi_filename).exists():
                source_midi = maestro_dir / midi_filename
            else:
                # Try to find by year and similar name
                year_dir = maestro_dir / year
                if year_dir.exists():
                    for file in year_dir.glob('*.mid*'):
                        if audio_id.replace('_wav', '') in file.stem:
                            source_midi = file
                            break
            
            total_files += 1
            
            # Skip if files not found
            if source_audio is None or source_midi is None:
                print(f"Could not find files for {audio_id}")
                continue
            
            # Resample audio if needed
            target_audio = output_dir / split / 'audio' / f"{audio_id}.wav"
            if args.sample_rate != 44100:
                audio, _ = librosa.load(source_audio, sr=args.sample_rate, mono=True)
                sf.write(target_audio, audio, args.sample_rate)
            else:
                shutil.copy(source_audio, target_audio)
            
            # Copy MIDI file
            target_midi = output_dir / split / 'midi' / f"{audio_id}.mid"
            shutil.copy(source_midi, target_midi)
            
            processed_files += 1
    
    print(f"Dataset preparation complete. Processed {processed_files}/{total_files} files.")
    print(f"Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare MAESTRO dataset for piano transcription')
    parser.add_argument('--maestro-dir', type=str, default='dataset/MAESTRO',
                      help='Path to MAESTRO dataset directory')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050, 44100],
                      help='Target sample rate for audio files')
    
    args = parser.parse_args()
    prepare_maestro_dataset(args)

if __name__ == "__main__":
    main() 
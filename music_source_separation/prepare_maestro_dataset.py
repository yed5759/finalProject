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

def create_output_dirs(output_dir):
    for split in ['train', 'val', 'test']:
        for subdir in ['audio', 'midi']:
            os.makedirs(output_dir / split / subdir, exist_ok=True)

def find_metadata_file(maestro_dir):
    # Load metadata
    metadata_path = maestro_dir / 'maestro-v3.0.0.json'

    # Try different metadata file locations/formats
    if not metadata_path.exists():
        metadata_path = maestro_dir / 'maestro-v3.0.0.csv'
    
    if not metadata_path.exists():
        # Try finding any CSV or JSON metadata files
        json_files = list(maestro_dir.glob('*.json'))
        csv_files = list(maestro_dir.glob('*.csv'))
        if json_files:
            metadata_path = json_files[0]
        elif csv_files:
            metadata_path = csv_files[0]
        else:
            raise FileNotFoundError(f"Could not find MAESTRO metadata in {maestro_dir}")
    return metadata_path

def load_metadata(metadata_path):
    # Load metadata based on file extension
    if metadata_path.suffix.lower() == '.json':
        with open(metadata_path, 'r') as f:
            metadata_json = json.load(f)

        # Handle different JSON formats
        if isinstance(metadata_json, dict):
            # Try different possible structures
            if 'items' in metadata_json:
                metadata = pd.DataFrame(metadata_json['items'])
            else:
                # The JSON might be a collection of records, convert directly to DataFrame
                metadata = pd.DataFrame(metadata_json)
        else:
            # If it's already a list
            metadata = pd.DataFrame(metadata_json)
    else:
        # CSV format
        metadata = pd.read_csv(metadata_path)

    # Check if we have the necessary columns
    required_cols = ['split', 'audio_filename', 'midi_filename']
    
    # Rename columns if needed (different versions might use different names)
    column_mapping = {
        'audio': 'audio_filename',
        'midi': 'midi_filename'
    }
    
    metadata = metadata.rename(columns=column_mapping)
    
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Metadata file missing required columns: {missing_cols}")
    
    return metadata

def group_files_by_split(metadata):
    train_files = metadata[metadata['split'] == 'train']
    validation_files = metadata[(metadata['split'] == 'validation') | (metadata['split'] == 'val')]
    test_files = metadata[metadata['split'] == 'test']
    return train_files, validation_files, test_files

def extract_audio_id(audio_filename):
    if isinstance(audio_filename, str):
        if audio_filename.endswith('.wav'):
            return os.path.basename(audio_filename).replace('.wav', '')
        return os.path.basename(audio_filename)
    return None

def find_source_file(maestro_dir, filename, audio_id, file_type='audio', year=None):
    # Try to find the file with exact path
    if isinstance(filename, str) and (maestro_dir / filename).exists():
        return maestro_dir / filename

    # Try to find by year and filename if year is available
    if year:
        year_dir = maestro_dir / year
        if year_dir.exists():
            pattern = '*.wav' if file_type == 'audio' else '*.mid*'
            for file in year_dir.glob(pattern):
                if audio_id in file.stem:
                    return file

    # If still not found, search all subdirectories
    pattern = '**/*.wav' if file_type == 'audio' else '**/*.mid*'
    for file in maestro_dir.glob(pattern):
        if audio_id in file.stem:
            return file

    return None

def process_file(source_audio, source_midi, output_dir, split, audio_id, sample_rate):
    # Resample audio if needed
    target_audio = output_dir / split / 'audio' / f"{audio_id}.wav"
    if sample_rate != 44100:
        try:
            audio, _ = librosa.load(source_audio, sr=sample_rate, mono=True)
            sf.write(target_audio, audio, sample_rate)
        except Exception as e:
            print(f"Error processing audio file {source_audio}: {e}")
            return False
    else:
        try:
            shutil.copy(source_audio, target_audio)
        except Exception as e:
            print(f"Error copying audio file {source_audio}: {e}")
            return False

    # Copy MIDI file
    target_midi = output_dir / split / 'midi' / f"{audio_id}.mid"
    try:
        shutil.copy(source_midi, target_midi)
    except Exception as e:
        print(f"Error copying MIDI file {source_midi}: {e}")
        return False

    return True


def prepare_maestro_dataset(args):
    """
    Prepare MAESTRO dataset for training, validation, and testing
    Works with existing dataset structure in dataset/MAESTRO
    """
    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    create_output_dirs(output_dir)

    metadata_path = find_metadata_file(maestro_dir)
    print(f"Using metadata file: {metadata_path}")

    metadata = load_metadata(metadata_path)

    # Group files by split
    train_files, validation_files, test_files = group_files_by_split(metadata)
    
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
            
            # Get year from filename or row data
            year = str(row['year']) if 'year' in row else None
            if not year and isinstance(audio_filename, str) and '/' in audio_filename:
                year = audio_filename.split('/')[0]
            
            # Extract audio ID from filename
            audio_id = extract_audio_id(audio_filename)
            if audio_id is None:
                print(f"Invalid audio_filename: {audio_filename}, skipping")
                continue

            source_audio = find_source_file(maestro_dir, audio_filename, audio_id, 'audio', year)
            source_midi = find_source_file(maestro_dir, midi_filename, audio_id.replace('_wav', ''), 'midi', year)

            total_files += 1
            
            # Skip if files not found
            if source_audio is None or source_midi is None:
                print(f"Could not find files for {audio_id}")
                if source_audio is None:
                    print(f"  Missing audio file: {audio_filename}")
                if source_midi is None:
                    print(f"  Missing MIDI file: {midi_filename}")
                continue
            
            if process_file(source_audio, source_midi, output_dir, split, audio_id, args.sample_rate):
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
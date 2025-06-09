# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pretty_midi
from pathlib import Path
from audio_features import load_or_extract_features, process_audio_file

# ===== Dataset for Training =====

class PianoTranscriptionDataset(Dataset):
    """
    Dataset for piano transcription
    """
    def __init__(self, audio_dir, midi_dir, features_dir=None, segment_length=None, 
                 sample_rate=16000, hop_length=512, n_cqt_bins=88):
        """
        Initialize dataset
        
        Args:
            audio_dir: Directory containing audio files (.wav)
            midi_dir: Directory containing MIDI files (.mid)
            features_dir: Directory for cached features (will be created if None)
            segment_length: Segment length in frames (if None, use full files)
            sample_rate: Audio sample rate for feature extraction
            hop_length: Hop length for feature extraction
            n_cqt_bins: Number of CQT bins
        """
        self.audio_dir = Path(audio_dir)
        self.midi_dir = Path(midi_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_cqt_bins = n_cqt_bins
        
        # Set up features directory
        if features_dir is None:
            # Create features directory parallel to audio directory
            self.features_dir = self.audio_dir.parent / 'features' / self.audio_dir.name
        else:
            self.features_dir = Path(features_dir)
        
        # Create features directory
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio and MIDI files
        self.audio_files = []
        self.midi_files = []
        
        for audio_file in sorted(self.audio_dir.glob('*.wav')):
            midi_file = self.midi_dir / f"{audio_file.stem}.mid"
            if midi_file.exists():
                self.audio_files.append(audio_file)
                self.midi_files.append(midi_file)
        
        print(f"Found {len(self.audio_files)} piano pieces")
        print(f"Features directory: {self.features_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        midi_file = self.midi_files[idx]
        
        try:
            # Define feature file path
            feature_file = self.features_dir / f"{audio_file.stem}_features.pkl"
            
            # Use the load_or_extract_features function
            features = load_or_extract_features(
                audio_path=audio_file,
                feature_path=feature_file,
                extractor_fn=process_audio_file,
                extractor_args={
                    'sample_rate': self.sample_rate,
                    'hop_length': self.hop_length,
                    'n_cqt_bins': self.n_cqt_bins
                },
                fallback_shape=(100, self.n_cqt_bins)
            )
            
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_file))
            piano_roll = midi_data.get_piano_roll(fs=self.sample_rate/self.hop_length)
            piano_roll = piano_roll[21:109]  # Keep only piano keys (A0 to C8)
            
            # Ensure piano roll matches feature length
            if piano_roll.shape[1] < len(features):
                # Pad piano roll if shorter
                pad_width = len(features) - piano_roll.shape[1]
                piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)))
            elif piano_roll.shape[1] > len(features):
                # Truncate piano roll if longer
                piano_roll = piano_roll[:, :len(features)]
            
            # Transpose piano roll to [time, pitch]
            piano_roll = piano_roll.T
            
            # Get a random segment if needed
            if self.segment_length and len(features) > self.segment_length:
                start = np.random.randint(0, len(features) - self.segment_length)
                features = features[start:start + self.segment_length]
                piano_roll = piano_roll[start:start + self.segment_length]
            
            # Convert to tensors
            features = torch.FloatTensor(features)
            piano_roll_binary = (piano_roll > 0).astype(np.float32)
            targets = torch.FloatTensor(piano_roll_binary)
            
            return features, targets
            
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return dummy data in case of error
            seq_len = self.segment_length if self.segment_length else 100
            return torch.zeros(seq_len, self.n_cqt_bins), torch.zeros(seq_len, 88)

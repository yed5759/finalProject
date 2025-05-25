import torch
import torch.nn as nn
import numpy as np
import math
import os
import librosa
import pretty_midi
from torch.utils.data import Dataset
from pathlib import Path
import pickle

# ===== Functions =====

def load_or_extract_features(audio_path, feature_path, extractor_fn, extractor_args, fallback_shape=(100, 88)):
    """
    Load features from cache if available, or extract and cache them.

    Args:
        audio_path (Path): Path to the input audio file
        feature_path (Path): Path to the cached .pkl feature file
        extractor_fn (callable): Function to extract features (e.g., process_audio_file)
        extractor_args (dict): Arguments to pass to extractor_fn
        fallback_shape (tuple): Shape of dummy fallback in case of failure

    Returns:
        np.ndarray: Feature matrix
    """
    if feature_path.exists():
        try:
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load cached features for {audio_path.name}: {e}")
            print("Recomputing features...")

    try:
        print(f"Extracting features for {audio_path.name}...")
        features = extractor_fn(audio_path, **extractor_args)

        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features cached to {feature_path}")
        return features
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {audio_path.name}: {e}")
        return np.zeros(fallback_shape, dtype=np.float32)

def process_audio_file(audio_file, sample_rate=16000, hop_length=512, n_cqt_bins=88, n_fft=2048):
    """
    Process audio file and extract CQT features
    
    Args:
        audio_file: Path to audio file
        sample_rate: Target sample rate
        hop_length: Hop length for CQT
        n_cqt_bins: Number of CQT bins (should match piano keys)
        n_fft: FFT size (not used for CQT but kept for compatibility)
    
    Returns:
        features: CQT features as numpy array [time, frequency]
    """
    # Load audio
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
    
    # Extract CQT features
    # We use n_cqt_bins to match piano keys (A0 to C8)
    cqt = librosa.cqt(
        audio, 
        sr=sample_rate, 
        hop_length=hop_length,
        n_bins=n_cqt_bins,
        fmin=librosa.note_to_hz('A0'),  # Lowest piano note (27.5 Hz)
        bins_per_octave=12  # 12 semitones per octave
    )
    
    # Convert to magnitude and transpose to [time, frequency]
    cqt_mag = np.abs(cqt).T
    
    # Apply log scaling for better dynamic range
    cqt_log = np.log1p(cqt_mag)
    
    return cqt_log.astype(np.float32)

# ===== Model Components =====

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, embed_dim]
        """
        seq_len = x.size(1)
        
        # Handle case where sequence is longer than max_len
        if seq_len > self.max_len:
            print(f"Warning: Sequence length {seq_len} is longer than maximum positional encoding length {self.max_len}.")
            print(f"Truncating or using chunked processing is recommended for very long sequences.")
            
            # Apply positional encoding to the maximum length we have
            x[:self.max_len] = x[:self.max_len] + self.pe
            
            # For the rest, just repeat the pattern (not ideal but better than crashing)
            # This means positions beyond max_len will start repeating their encoding pattern
            for pos in range(self.max_len, seq_len, self.max_len):
                end_pos = min(pos + self.max_len, seq_len)
                x[:, pos:end_pos, :] += self.pe[:, :end_pos - pos,:]
        else:
            # Standard case: sequence length is within max_len
            x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)

class PianoTransformer(nn.Module):
    def __init__(self, n_cqt_bins=88, hidden_dim=256, num_heads=4, num_layers=3, 
                 dropout=0.1, max_len=1000):
        super(PianoTransformer, self).__init__()
        
        input_dim = n_cqt_bins # CQT bins
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True  # Use batch_first=True for better GPU performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers for onset, offset, and velocity
        self.note_layer = nn.Linear(hidden_dim, 88)  # Only note presence
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters of the model with bias initialization for output layers
        to ensure some predictions are active from the start
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize output layer biases to small positive values
        # This gives a small positive bias to the logits, which helps with the initial predictions
        # beacause the notes presences per frame are sparse (mostly 0) we need higher constant. 
        # changed it from 0.1 to 1.0. chatGPT say we should try also 2.0, 3.0 ...
        nn.init.constant_(self.note_layer.bias, 1.0)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        
        Args:
            src: Source tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Mask for source tensor
            
        Returns:
            note_presence: Note presence logits [batch_size, seq_len, 88]
            velocity: Velocity logits [batch_size, seq_len, 88]
        """
        # With batch_first=True, we don't need to transpose the input
        
        # Embed input
        src = self.input_norm(self.embedding(src) * math.sqrt(self.hidden_dim))
        
        # Add positional encoding - need to adapt this for batch_first=True
        # The positional encoding expects [seq_len, batch_size, hidden_dim]
        # So we transpose, add positional encoding, and transpose back
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Apply output layers to get logits
        note_presence = self.note_layer(output)
        
        # Return logits (without sigmoid) for loss function
        return note_presence

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

# ===== MIDI Utilities =====

def create_midi_from_predictions(note_presence, output_file, threshold=0.5, tempo=120.0, velocity=80):
    """
    Convert model predictions to a MIDI file using note presence and velocity representation
    
    Args:
        predictions: Model predictions, can be:
                    - A single array with shape (time_steps, pitch_bins*3)
                    - A tuple of (note_presence, velocity) each with shape (time_steps, 88)
        output_file: Output MIDI file path
        threshold: Threshold for detecting note onsets
        tempo: Tempo of the output MIDI file in BPM
    """
    # Create MIDI file
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Convert to probabilities if needed, using sigmoid
    if np.max(np.abs(note_presence)) > 1.0:
        note_presence = 1.0 / (1.0 + np.exp(-note_presence))

    # Process each time step
    for t in range(len(note_presence)):
        # Find active notes
        active_notes = np.where(note_presence[t] > threshold)[0]
        
        for note in active_notes:
            # Create MIDI note
            vel = int(velocity[t, note] * 127) if isinstance(velocity, (np.ndarray, torch.Tensor)) else velocity
            midi_note = pretty_midi.Note(
                velocity=vel,  # Scale to MIDI velocity range
                pitch=note + 21,  # Convert to MIDI pitch (A0 = 21)
                start=t * 0.01,  # 10ms per frame
                end=(t + 1) * 0.01
            )
            piano.notes.append(midi_note)
    
    # Add piano to MIDI file and save
    midi_data.instruments.append(piano)
    midi_data.write(output_file)
    
    return midi_data

# ===== Transcription System =====

class PianoTranscriptionSystem:
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        
        # Create model
        self.model = PianoTransformer(n_cqt_bins=88).to(device)
        
        # Load pretrained model if provided
        if model_path:
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("No model provided, using random initialization")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def transcribe(self, audio_file, output_midi_file, sample_rate=16000, 
                   n_fft=2048, hop_length=512, n_cqt_bins=88):

        print(f"Transcribing {audio_file} to {output_midi_file}")
        
        # Extract features
        features = process_audio_file(
            audio_file,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_cqt_bins=n_cqt_bins,
            n_fft=n_fft
        )
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Run model
        with torch.no_grad():
            note_presence = self.model(features)
        
        midi_data = create_midi_from_predictions(
            note_presence,
            output_file=output_midi_file,
            threshold=0.5,
            tempo=120.0
        )
        
        return output_midi_file 

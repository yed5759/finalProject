import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import pretty_midi
from torch.utils.data import Dataset
from pathlib import Path
import pickle

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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, embed_dim]
        """
        seq_len = x.size(0)
        
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
                pattern_len = end_pos - pos
                x[pos:end_pos] = x[pos:end_pos] + self.pe[:pattern_len]
        else:
            # Standard case: sequence length is within max_len
            x = x + self.pe[:seq_len]
        
        return self.dropout(x)

class PianoTransformer(nn.Module):
    """
    Transformer model for piano transcription with separate onset, offset, and velocity predictions
    """
    def __init__(self, n_cqt_bins=88, hidden_dim=256, num_heads=4, num_layers=3, 
                 dropout=0.1, max_len=1000):
        super(PianoTransformer, self).__init__()
        
        input_dim = n_cqt_bins + 1  # CQT bins + velocity features
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
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
        self.note_layer = nn.Linear(hidden_dim, 88) #For note presence
        self.velocity_layer = nn.Linear(hidden_dim, 88) #For velocity
        
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
        nn.init.constant_(self.note_layer.bias, 0.1)
        nn.init.constant_(self.velocity_layer.bias, 0.1)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        
        Args:
            src: Source tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Mask for source tensor
            
        Returns:
            onset_logits: Onset logits [batch_size, seq_len, 88]
            offset_logits: Offset logits [batch_size, seq_len, 88]
            velocity_logits: Velocity logits [batch_size, seq_len, 88]
        """
        # With batch_first=True, we don't need to transpose the input
        
        # Embed input
        src = self.embedding(src) * np.sqrt(self.hidden_dim)
        
        # Add positional encoding - need to adapt this for batch_first=True
        # The positional encoding expects [seq_len, batch_size, hidden_dim]
        # So we transpose, add positional encoding, and transpose back
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Apply output layers to get logits
        note_presence = self.note_layer(output)
        velocity = self.velocity_layer(output)
        
        # Return logits (without sigmoid) for loss function
        return note_presence, velocity

# ===== Audio Processing =====

#Move to preprocess

def load_audio(file_path, sample_rate=16000):
    """
    Load audio file and resample if necessary
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, sr

def extract_cqt_features(audio, sr, hop_length=512, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('A0')):
    """
    Extract Constant-Q Transform (CQT) features from audio
    """
    # Extract CQT
    cqt = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin
    )
    
    # Convert to log scale
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    return cqt_spectrogram

def extract_onset_features(audio, sr, n_fft=2048, hop_length=512):
    """
    Extract onset detection features from audio
    """
    # Compute onset strength
    onset_env = librosa.onset.onset_strength(
        y=audio, 
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, 
        sr=sr,
        hop_length=hop_length,
        backtrack=True
    )
    
    # Create onset indicator array
    onsets = np.zeros(len(onset_env))
    onsets[onset_frames] = 1
    
    return onsets, onset_env

def normalize_features(features):
    """
    Normalize features to range [0, 1]
    """
    min_val = np.min(features)
    max_val = np.max(features)
    
    normalized_features = (features - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_features

def preprocess_audio(file_path, sample_rate=16000, n_fft=2048, hop_length=512, 
                    n_cqt_bins=84):
    """
    Full preprocessing pipeline for audio files using only CQT
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (16000 or 22050 recommended)
        n_fft: FFT window size
        hop_length: Hop length for FFT
        n_cqt_bins: Number of CQT bins
        
    Returns:
        features: Preprocessed features ready for model input
        audio_length: Length of audio in seconds
    """
    # Load audio
    audio, sr = load_audio(file_path, sample_rate)
    
    # Calculate audio length in seconds
    audio_length = len(audio) / sr
    
    # Extract onset information (for onset detection)
    onsets, onset_env = extract_onset_features(audio, sr, n_fft, hop_length)
    
    # Extract CQT features
    cqt_features = extract_cqt_features(audio, sr, hop_length, n_cqt_bins)
    
    # Compute offset detection features (negative of onset detection)
    offset_env = -np.diff(onset_env, append=0)
    offset_env = np.maximum(0, offset_env)
    
    # Compute velocity information (using RMSE as a proxy for velocity)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    velocity = librosa.util.normalize(rms) 
    
    # Stack all features
    features = np.vstack([
        cqt_features,               # CQT bins
        onset_env.reshape(1, -1),   # Onset detection
        offset_env.reshape(1, -1),  # Offset detection
        velocity.reshape(1, -1)     # Velocity information
    ])
    
    # Normalize features
    features = normalize_features(features)
    
    return features, audio_length

# Function to process an audio file and extract features
def process_audio_file(file_path, sample_rate=16000, hop_length=512, n_cqt_bins=88, n_fft=2048):
    """
    Process audio file and extract features for model input
    
    Args:
        file_path: Path to audio file
        sample_rate: Sample rate
        hop_length: Hop length for feature extraction
        n_cqt_bins: Number of CQT bins
        n_fft: FFT window size
        
    Returns:
        features: Audio features (CQT, onset, offset, velocity)
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    # Extract CQT features
    cqt = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_cqt_bins,
        bins_per_octave=12,
        fmin=librosa.note_to_hz('A0')
    )
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Extract onset features
    onset_env = librosa.onset.onset_strength(
        y=audio, 
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Compute offset features
    offset_env = -np.diff(onset_env, append=0)
    offset_env = np.maximum(0, offset_env)
    
    # Compute velocity features
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    velocity = librosa.util.normalize(rms) 
    
    # Stack features
    features = np.vstack([
        cqt_spectrogram,          # CQT bins
        onset_env.reshape(1, -1), # Onset detection
        offset_env.reshape(1, -1), # Offset detection
        velocity.reshape(1, -1)   # Velocity information
    ])
    
    # Normalize features
    min_val = np.min(features)
    max_val = np.max(features)
    features = (features - min_val) / (max_val - min_val + 1e-8)
    
    # Transpose to (time, features)
    features = features.T
    
    return features

# Function to convert MIDI to piano roll (onset, offset, velocity)
def midi_to_piano_roll(midi_data, hop_length=512, sample_rate=16000, roll_length=None):
    """
    Convert MIDI to piano roll with onset, offset and velocity information
    
    Args:
        midi_data: PrettyMIDI object
        hop_length: Hop length (in samples)
        sample_rate: Sample rate
        roll_length: Length of piano roll (in frames)
        
    Returns:
        onsets: Binary onset matrix (frames x 88 keys)
        offsets: Binary offset matrix (frames x 88 keys)
        velocities: Normalized velocity matrix (frames x 88 keys)
    """
    # Get max time
    if roll_length is None:
        max_time = midi_data.get_end_time()
        roll_length = int(max_time * sample_rate / hop_length) + 1
    
    # Create empty matrices
    onsets = np.zeros((roll_length, 88))
    offsets = np.zeros((roll_length, 88))
    velocities = np.zeros((roll_length, 88))
    
    # Frame timing (seconds per frame)
    frame_time = hop_length / sample_rate
    
    # Process each note
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
            
        for note in instrument.notes:
            # Skip notes outside piano range
            if note.pitch < 21 or note.pitch > 108:
                continue
                
            # Convert pitch to piano key (0-87)
            key = note.pitch - 21
            
            # Convert time to frame
            start_frame = int(note.start / frame_time)
            end_frame = int(note.end / frame_time)
            
            if start_frame >= roll_length or end_frame < 0:
                continue
                
            # Ensure valid indices
            start_frame = max(0, start_frame)
            end_frame = min(roll_length - 1, end_frame)
            
            # Mark onset and offset
            onsets[start_frame, key] = 1.0
            if end_frame < roll_length:
                offsets[end_frame, key] = 1.0
                
            # Set velocity
            normalized_velocity = note.velocity / 127.0
            velocities[start_frame:end_frame+1, key] = normalized_velocity
    
    return onsets, offsets, velocities

# Function to convert piano roll to MIDI file
def notes_to_midi(onsets, offsets, velocities, hop_length=512, sample_rate=16000):
    """
    Convert piano roll matrices to MIDI file
    
    Args:
        onsets: Binary onset matrix (frames x 88 keys)
        offsets: Binary offset matrix (frames x 88 keys)
        velocities: Velocity matrix (frames x 88 keys)
        hop_length: Hop length in samples
        sample_rate: Sample rate
        
    Returns:
        midi_obj: PrettyMIDI object
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Frame timing
    frame_time = hop_length / sample_rate
    
    # Process each piano key
    for key in range(88):
        midi_pitch = key + 21  # Convert to MIDI pitch (A0 = 21)
        
        # Find onset frames
        onset_frames = np.where(onsets[:, key] > 0.5)[0]
        
        for onset_frame in onset_frames:
            # Find the next offset
            offset_frames = np.where(offsets[onset_frame:, key] > 0.5)[0]
            
            if len(offset_frames) > 0:
                # Offset is relative to onset_frame, so add it back
                offset_frame = offset_frames[0] + onset_frame
            else:
                # If no offset found, set to end of track
                offset_frame = len(onsets) - 1
            
            # Get start and end times in seconds
            start_time = onset_frame * frame_time
            end_time = offset_frame * frame_time
            
            # Ensure note has minimum duration
            if end_time <= start_time:
                end_time = start_time + 0.1  # 100ms minimum
            
            # Get velocity (use onset frame's velocity)
            velocity = int(min(max(velocities[onset_frame, key] * 127, 1), 127))
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=start_time,
                end=end_time
            )
            
            # Add to instrument
            piano.notes.append(note)
    
    # Add piano to MIDI object
    midi.instruments.append(piano)
    return midi

# ===== Dataset for Training =====

class PianoTranscriptionDataset(Dataset):
    """
    Dataset for piano transcription
    """
    def __init__(self, features_dir, midi_dir, segment_length=None):
        """
        Initialize dataset
        
        Args:
            features_dir: Directory containing preprocessed features
            midi_dir: Directory containing MIDI files
            segment_length: Segment length in frames (if None, use full files)
        """
        self.features_dir = Path(features_dir)
        self.midi_dir = Path(midi_dir)
        self.segment_length = segment_length
        
        # Find all feature and MIDI files
        self.feature_files = []
        self.midi_files = []
        
        for feature_file in sorted(self.features_dir.glob('*_features.pkl')):
            midi_file = self.midi_dir / f"{feature_file.stem.replace('_features', '')}.mid"
            if midi_file.exists():
                self.feature_files.append(feature_file)
                self.midi_files.append(midi_file)
        
        print(f"Found {len(self.feature_files)} piano pieces")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        midi_file = self.midi_files[idx]
        
        try:
            # Load preprocessed features
            with open(feature_file, 'rb') as f:
                features = pickle.load(f)
            
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_file))
            piano_roll = midi_data.get_piano_roll(fs=16000/512)
            piano_roll = piano_roll[21:109]  # Keep only piano keys
            
            # Get a random segment if needed
            if self.segment_length and len(features) > self.segment_length:
                start = np.random.randint(0, len(features) - self.segment_length)
                features = features[start:start + self.segment_length]
                piano_roll = piano_roll[:, start:start + self.segment_length]
            
            # Convert to tensors
            features = torch.FloatTensor(features)
            piano_roll = torch.FloatTensor(piano_roll.T)
            
            return features, piano_roll
            
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
            # Return dummy data in case of error
            return torch.zeros(100, 89), torch.zeros(100, 88)

def collate_fn(batch):
    """
    Collate function to make all sequences the same length
    
    Args:
        batch: List of (audio_features, midi_target) pairs
        
    Returns:
        Batch of tensors with same length
    """
    # Find shortest sequence
    min_length = min(x[0].shape[0] for x in batch)
    
    # Stack tensors, truncating to shortest length
    audio_features = torch.stack([x[0][:min_length] for x in batch])
    midi_target = torch.stack([x[1][:min_length] for x in batch])
    
    return audio_features, midi_target

# ===== MIDI Utilities =====

def create_midi_from_predictions(predictions, output_file, onset_threshold=0.5, tempo=120.0):
    """
    Convert model predictions to a MIDI file using note presence and velocity representation
    
    Args:
        predictions: Model predictions, can be:
                    - A single array with shape (time_steps, pitch_bins*3)
                    - A tuple of (note_presence, velocity) each with shape (time_steps, 88)
        output_file: Output MIDI file path
        onset_threshold: Threshold for detecting note onsets
        tempo: Tempo of the output MIDI file in BPM
    """
    # Create MIDI file
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Get predictions
    note_presence, velocity = predictions
    
    # Convert to probabilities if needed, using sigmoid
    if np.max(np.abs(note_presence)) > 1.0:
        note_presence = 1.0 / (1.0 + np.exp(-note_presence))

    # Process each time step
    for t in range(len(note_presence)):
        # Find active notes
        active_notes = np.where(note_presence[t] > threshold)[0]
        
        for note in active_notes:
            # Create MIDI note
            midi_note = pretty_midi.Note(
                velocity=int(velocity[t, note] * 127),  # Scale to MIDI velocity range
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
                   n_fft=2048, hop_length=512, n_cqt_bins=84):

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
            note_presence_logits, velocity = self.model(features)
        note_presence = note_presence_logits.squeeze(0).cpu().numpy()
        velocity = velocity_logits.squeeze(0).cpu().numpy()
        
        midi_data = create_midi_from_predictions(
            (note_presence, velocity),
            output_file=output_midi_file,
            hop_time=hop_length / sample_rate  # Convert hop length to time
        )
        
        return output_midi_file 
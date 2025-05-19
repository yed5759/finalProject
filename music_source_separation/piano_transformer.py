import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import pretty_midi
from torch.utils.data import Dataset
from pathlib import Path

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
    def __init__(self, n_cqt_bins=88, hidden_dim=512, num_heads=8, num_layers=6, 
                 dropout=0.1, max_len=5000):
        super(PianoTransformer, self).__init__()
        
        input_dim = n_cqt_bins + 3  # CQT bins + onset + offset + velocity features
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True  # Use batch_first=True for better GPU performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers for onset, offset, and velocity
        self.onset_layer = nn.Linear(hidden_dim, 88)
        self.offset_layer = nn.Linear(hidden_dim, 88)
        self.velocity_layer = nn.Linear(hidden_dim, 88)
        
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
        nn.init.constant_(self.onset_layer.bias, 0.1)
        nn.init.constant_(self.offset_layer.bias, 0.1)
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
        onset_logits = self.onset_layer(output)
        offset_logits = self.offset_layer(output)
        velocity_logits = self.velocity_layer(output)
        
        # Return logits (without sigmoid) for loss function
        return onset_logits, offset_logits, velocity_logits

# ===== Audio Processing =====

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
    def __init__(self, audio_dir, midi_dir, segment_length=None, hop_length=512, 
                 sample_rate=16000, n_cqt_bins=88, random_offset=True):
        """
        Initialize dataset
        
        Args:
            audio_dir: Directory containing audio files
            midi_dir: Directory containing MIDI files
            segment_length: Segment length in seconds (if None, use full files)
            hop_length: Hop length for feature extraction
            sample_rate: Audio sample rate
            n_cqt_bins: Number of CQT bins
            random_offset: Whether to use random offset for training
        """
        self.audio_dir = Path(audio_dir)
        self.midi_dir = Path(midi_dir)
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_cqt_bins = n_cqt_bins
        self.random_offset = random_offset
        
        # Frame length if segment_length is specified
        self.segment_frames = int(segment_length * sample_rate / hop_length) if segment_length else None
        
        # Find all audio files
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        
        # Check for matching MIDI files
        self.midi_files = []
        valid_audio_files = []
        
        for audio_file in self.audio_files:
            midi_file = self.midi_dir / f"{audio_file.stem}.mid"
            if midi_file.exists():
                valid_audio_files.append(audio_file)
                self.midi_files.append(midi_file)
        
        self.audio_files = valid_audio_files
        
        print(f"Found {len(self.audio_files)} audio files with matching MIDI")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        midi_file = self.midi_files[idx]
        
        # Process audio file
        try:
            # Extract features
            audio_features = process_audio_file(
                audio_file,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                n_cqt_bins=self.n_cqt_bins
            )
            
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_file))
            
            # Convert to piano roll
            onsets, offsets, velocities = midi_to_piano_roll(
                midi_data,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
                roll_length=len(audio_features)
            )
            
            # Select segment if needed
            if self.segment_frames and len(audio_features) > self.segment_frames:
                if self.random_offset:
                    # Random offset for training
                    max_offset = len(audio_features) - self.segment_frames
                    offset = np.random.randint(0, max_offset)
                else:
                    # Use middle segment for validation
                    offset = (len(audio_features) - self.segment_frames) // 2
                
                # Extract segment
                audio_features = audio_features[offset:offset+self.segment_frames]
                onsets = onsets[offset:offset+self.segment_frames]
                offsets = offsets[offset:offset+self.segment_frames]
                velocities = velocities[offset:offset+self.segment_frames]
            
            # Convert to tensors
            audio_features = torch.FloatTensor(audio_features)
            onsets = torch.FloatTensor(onsets)
            offsets = torch.FloatTensor(offsets)
            velocities = torch.FloatTensor(velocities)
            
            return audio_features, onsets, offsets, velocities
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            # Return a very small dummy sample in case of error
            dummy_length = 100
            return (
                torch.zeros(dummy_length, self.n_cqt_bins + 3),
                torch.zeros(dummy_length, 88),
                torch.zeros(dummy_length, 88),
                torch.zeros(dummy_length, 88)
            )

def collate_fn(batch):
    """
    Collate function for variable length sequences
    
    Args:
        batch: List of tuples (audio_features, onsets, offsets, velocities)
        
    Returns:
        Padded batch
    """
    # Find the shortest sequence in the batch
    min_length = min(x[0].shape[0] for x in batch)
    
    # Truncate all sequences to the same length
    audio_features = torch.stack([x[0][:min_length] for x in batch])
    onsets = torch.stack([x[1][:min_length] for x in batch])
    offsets = torch.stack([x[2][:min_length] for x in batch])
    velocities = torch.stack([x[3][:min_length] for x in batch])
    
    return audio_features, onsets, offsets, velocities

# ===== MIDI Utilities =====

def create_midi_from_predictions(predictions, output_file, onset_threshold=0.5, 
                                offset_threshold=0.5, velocity_scale=100,
                                min_note_length=3, tempo=120.0, hop_time=0.01):
    """
    Convert model predictions to a MIDI file using onset-offset-velocity representation
    
    Args:
        predictions: Model predictions, can be:
                    - A single array with shape (time_steps, pitch_bins*3)
                    - A tuple of (onset_preds, offset_preds, velocity_preds) each with shape (time_steps, 88)
        output_file: Output MIDI file path
        onset_threshold: Threshold for detecting note onsets
        offset_threshold: Threshold for detecting note offsets
        velocity_scale: Scaling factor for velocity (0-127)
        min_note_length: Minimum note length in frames
        tempo: Tempo of the output MIDI file in BPM
        hop_time: Time between consecutive frames in seconds
    """
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create a piano instrument
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
    
    # Handle different input formats
    num_pitches = 88
    
    if isinstance(predictions, tuple) and len(predictions) == 3:
        # Separate predictions for onset, offset, velocity
        onset_predictions, offset_predictions, velocity_predictions = predictions
        
        # Apply sigmoid to convert logits to probabilities if needed
        if isinstance(onset_predictions, np.ndarray) and np.max(np.abs(onset_predictions)) > 1.0:
            onset_predictions = 1.0 / (1.0 + np.exp(-onset_predictions))  # Apply sigmoid
        
        if isinstance(offset_predictions, np.ndarray) and np.max(np.abs(offset_predictions)) > 1.0:
            offset_predictions = 1.0 / (1.0 + np.exp(-offset_predictions))  # Apply sigmoid
            
        if isinstance(velocity_predictions, np.ndarray) and np.max(np.abs(velocity_predictions)) > 1.0:
            velocity_predictions = 1.0 / (1.0 + np.exp(-velocity_predictions))  # Apply sigmoid
    
    elif isinstance(predictions, np.ndarray):
        if predictions.shape[1] == num_pitches * 3:
            # Multi-task output in single array: [onset, offset, velocity] for each pitch
            onset_predictions = predictions[:, :num_pitches]
            offset_predictions = predictions[:, num_pitches:2*num_pitches]
            velocity_predictions = predictions[:, 2*num_pitches:3*num_pitches]
            
            # Apply sigmoid to convert logits to probabilities if needed
            if np.max(np.abs(onset_predictions)) > 1.0:
                onset_predictions = 1.0 / (1.0 + np.exp(-onset_predictions))
            
            if np.max(np.abs(offset_predictions)) > 1.0:
                offset_predictions = 1.0 / (1.0 + np.exp(-offset_predictions))
                
            if np.max(np.abs(velocity_predictions)) > 1.0:
                velocity_predictions = 1.0 / (1.0 + np.exp(-velocity_predictions))
        else:
            # Single-task output: treat as frame activation and derive onset/offset
            frame_predictions = predictions
            
            # Apply sigmoid if needed
            if np.max(np.abs(frame_predictions)) > 1.0:
                frame_predictions = 1.0 / (1.0 + np.exp(-frame_predictions))
            
            # Derive onset predictions
            onset_predictions = np.zeros_like(frame_predictions)
            onset_predictions[1:] = np.maximum(0, frame_predictions[1:] - frame_predictions[:-1])
            
            # Derive offset predictions
            offset_predictions = np.zeros_like(frame_predictions)
            offset_predictions[:-1] = np.maximum(0, frame_predictions[:-1] - frame_predictions[1:])
            
            # Use frame activations as velocity
            velocity_predictions = frame_predictions
    else:
        raise ValueError("Predictions must be a tuple of (onset, offset, velocity) or a single array")
    
    # Track active notes for each pitch
    active_notes = {}  # {pitch: (start_time, velocity)}
    
    # Process each frame for note events
    for frame in range(len(onset_predictions)):
        frame_time = frame * hop_time
        
        # Check for note onsets
        for pitch in range(num_pitches):
            # Note onset detected
            if onset_predictions[frame, pitch] > onset_threshold:
                # Store onset information
                velocity_value = velocity_predictions[frame, pitch]
                active_notes[pitch] = (frame_time, velocity_value)
            
            # Note offset detected
            elif pitch in active_notes and offset_predictions[frame, pitch] > offset_threshold:
                # Get onset information
                start_time, velocity_value = active_notes[pitch]
                
                # Calculate duration
                duration = frame_time - start_time
                
                # Only add note if it's long enough
                if duration >= min_note_length * hop_time:
                    # Scale velocity to MIDI range (0-127)
                    velocity = max(1, min(127, int(velocity_value * velocity_scale)))
                    
                    # Create MIDI note
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch + 21,  # MIDI pitches start at 21 (A0)
                        start=start_time,
                        end=frame_time
                    )
                    
                    # Add to piano
                    piano.notes.append(note)
                
                # Remove from active notes
                del active_notes[pitch]
    
    # Handle notes that are still active at the end
    end_time = len(onset_predictions) * hop_time
    for pitch, (start_time, velocity_value) in active_notes.items():
        # Calculate duration
        duration = end_time - start_time
        
        # Only add note if it's long enough
        if duration >= min_note_length * hop_time:
            # Scale velocity to MIDI range (0-127)
            velocity = max(1, min(127, int(velocity_value * velocity_scale)))
            
            # Create MIDI note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch + 21,  # MIDI pitches start at 21 (A0)
                start=start_time,
                end=end_time
            )
            
            # Add to piano
            piano.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(piano)
    
    # Write out the MIDI data
    midi_data.write(output_file)
    
    return midi_data

# ===== Transcription System =====

class PianoTranscriptionSystem:
    """
    Piano transcription system using the trained model
    """
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the transcription system
        
        Args:
            model_path: Path to the pretrained model (if None, a random model will be created)
            device: Device to run the model on ('cuda' or 'cpu')
        """
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
        """
        Transcribe audio file to MIDI
        
        Args:
            audio_file: Path to audio file
            output_midi_file: Path to output MIDI file
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            n_cqt_bins: Number of CQT bins
            
        Returns:
            Path to output MIDI file
        """
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
            onset_logits, offset_logits, velocity_logits = self.model(features)
        
        # Convert logits to numpy arrays
        onset_logits = onset_logits.squeeze(0).cpu().numpy()
        offset_logits = offset_logits.squeeze(0).cpu().numpy()
        velocity_logits = velocity_logits.squeeze(0).cpu().numpy()
        
        # Create MIDI file
        midi_data = create_midi_from_predictions(
            (onset_logits, offset_logits, velocity_logits),
            output_file=output_midi_file,
            hop_time=hop_length / sample_rate  # Convert hop length to time
        )
        
        return output_midi_file 
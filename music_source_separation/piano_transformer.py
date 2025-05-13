import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import pretty_midi

# ===== Model Components =====

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PianoTransformer(nn.Module):
    """
    Transformer model for piano transcription
    """
    def __init__(self, input_dim=128, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, output_dim=88, dropout=0.1, max_len=5000):
        super(PianoTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters of the model
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        
        Args:
            src: Source tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Mask for source tensor
            
        Returns:
            output: Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Transpose to [seq_len, batch_size, input_dim]
        src = src.transpose(0, 1)
        
        # Embed and add positional encoding
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Output layer
        output = self.output_layer(output)
        
        # Transpose back to [batch_size, seq_len, output_dim]
        output = output.transpose(0, 1)
        
        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        
        return output

# ===== Audio Processing =====

def load_audio(file_path, sample_rate=16000):
    """
    Load audio file and resample if necessary
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, sr

def extract_mel_features(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extract mel spectrogram features from audio
    """
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram

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
                    feature_type='mel', n_mels=128, n_cqt_bins=84):
    """
    Full preprocessing pipeline for audio files
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (16000 or 22050 recommended)
        n_fft: FFT window size
        hop_length: Hop length for FFT
        feature_type: Type of features to extract ('mel', 'cqt', or 'both')
        n_mels: Number of mel bands
        n_cqt_bins: Number of CQT bins
        
    Returns:
        features: Preprocessed features ready for model input
        audio_length: Length of audio in seconds
    """
    # Load audio
    audio, sr = load_audio(file_path, sample_rate)
    
    # Calculate audio length in seconds
    audio_length = len(audio) / sr
    
    # Extract onset information
    onsets, onset_env = extract_onset_features(audio, sr, n_fft, hop_length)
    
    # Extract spectral features based on specified type
    if feature_type == 'mel':
        spectral_features = extract_mel_features(audio, sr, n_fft, hop_length, n_mels)
        features = spectral_features
    elif feature_type == 'cqt':
        spectral_features = extract_cqt_features(audio, sr, hop_length, n_cqt_bins)
        features = spectral_features
    elif feature_type == 'both':
        mel_features = extract_mel_features(audio, sr, n_fft, hop_length, n_mels)
        cqt_features = extract_cqt_features(audio, sr, hop_length, n_cqt_bins)
        
        # Stack the features
        features = np.vstack([mel_features, cqt_features])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Use 'mel', 'cqt', or 'both'")
    
    # Add onset information as additional row in features
    onset_env = onset_env.reshape(1, -1)
    if len(onset_env[0]) < features.shape[1]:
        # Pad if necessary
        padded_onset = np.pad(onset_env, ((0, 0), (0, features.shape[1] - len(onset_env[0]))))
        features = np.vstack([features, padded_onset])
    elif len(onset_env[0]) > features.shape[1]:
        # Truncate if necessary
        onset_env = onset_env[:, :features.shape[1]]
        features = np.vstack([features, onset_env])
    else:
        features = np.vstack([features, onset_env])
    
    # Normalize features
    features = normalize_features(features)
    
    return features, audio_length

# ===== MIDI Utilities =====

def create_midi_from_predictions(predictions, output_file, onset_threshold=0.5, 
                                offset_threshold=0.3, velocity_scale=100,
                                min_note_length=3, tempo=120.0, hop_time=0.01):
    """
    Convert model predictions to a MIDI file with onset, offset and velocity
    
    Args:
        predictions: Model predictions with shape (time_steps, pitch_bins)
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
    
    # Split predictions if we have multi-task outputs
    if predictions.shape[1] > 88 and predictions.shape[1] % 88 == 0:
        num_outputs = predictions.shape[1] // 88
        # Assume first 88 are frame activations, second 88 are onsets, third could be offsets or velocities
        frame_predictions = predictions[:, :88]
        
        if num_outputs >= 2:
            onset_predictions = predictions[:, 88:176]
        else:
            # Calculate the difference to detect onsets
            onset_predictions = np.zeros_like(frame_predictions)
            onset_predictions[1:] = frame_predictions[1:] - frame_predictions[:-1]
            onset_predictions = np.maximum(0, onset_predictions)
        
        if num_outputs >= 3:
            offset_predictions = predictions[:, 176:264]
        else:
            # Calculate the negative difference to detect offsets
            offset_predictions = np.zeros_like(frame_predictions)
            offset_predictions[:-1] = frame_predictions[:-1] - frame_predictions[1:]
            offset_predictions = np.maximum(0, offset_predictions)
        
        if num_outputs >= 4:
            velocity_predictions = predictions[:, 264:352]
        else:
            # Use frame activation as a proxy for velocity
            velocity_predictions = frame_predictions
    else:
        # Just frame activations, derive everything else
        frame_predictions = predictions
        
        # Calculate the difference to detect onsets
        onset_predictions = np.zeros_like(frame_predictions)
        onset_predictions[1:] = frame_predictions[1:] - frame_predictions[:-1]
        onset_predictions = np.maximum(0, onset_predictions)
        
        # Use frame activation as a proxy for velocity
        velocity_predictions = frame_predictions
    
    # Iterate through time steps and pitches
    for pitch in range(88):  # 88 keys on a piano
        note_start = None
        max_velocity = 0
        
        # Go through all time steps
        for frame in range(len(frame_predictions)):
            # Note onset detected
            if onset_predictions[frame, pitch] > onset_threshold and frame_predictions[frame, pitch] > 0.1:
                if note_start is None:
                    note_start = frame
                    max_velocity = velocity_predictions[frame, pitch]
                else:
                    # If we detect another onset while a note is active, treat it as a new note
                    # Store the current note first
                    note_end = frame
                    if note_end - note_start >= min_note_length:
                        add_note_to_piano(piano, note_start, note_end, pitch, max_velocity, hop_time, velocity_scale)
                    
                    # Start a new note
                    note_start = frame
                    max_velocity = velocity_predictions[frame, pitch]
            
            # Update maximum velocity if needed
            elif note_start is not None and velocity_predictions[frame, pitch] > max_velocity:
                max_velocity = velocity_predictions[frame, pitch]
            
            # Note offset detected
            elif note_start is not None and (frame_predictions[frame, pitch] < offset_threshold or 
                                           (num_outputs >= 3 and offset_predictions[frame, pitch] > offset_threshold)):
                note_end = frame
                if note_end - note_start >= min_note_length:
                    add_note_to_piano(piano, note_start, note_end, pitch, max_velocity, hop_time, velocity_scale)
                note_start = None
                max_velocity = 0
        
        # Handle notes that are active until the end
        if note_start is not None:
            note_end = len(frame_predictions)
            if note_end - note_start >= min_note_length:
                add_note_to_piano(piano, note_start, note_end, pitch, max_velocity, hop_time, velocity_scale)
    
    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(piano)
    
    # Write out the MIDI data
    midi_data.write(output_file)
    
    return midi_data

def add_note_to_piano(piano, start_frame, end_frame, pitch, velocity_value, hop_time, velocity_scale):
    """Helper function to add a note to the piano instrument"""
    # Calculate start and end times in seconds
    start_time = start_frame * hop_time
    end_time = end_frame * hop_time
    
    # MIDI pitches typically start at 21 (A0)
    midi_pitch = pitch + 21
    
    # Scale velocity to MIDI range (0-127)
    velocity = max(1, min(127, int(velocity_value * velocity_scale)))
    
    # Create a Note object
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=midi_pitch,
        start=start_time,
        end=end_time
    )
    
    # Add it to our instrument
    piano.notes.append(note)

# ===== Transcription System =====

class PianoTranscriptionSystem:
    """
    Full piano transcription system
    """
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the system
        
        Args:
            model_path: Path to the trained model
            device: Device to run the model on
        """
        self.device = device
        
        # Create model
        self.model = PianoTransformer().to(device)
        
        # Load pretrained model if path is provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
    
    def transcribe(self, audio_file, output_midi_file, sample_rate=16000, 
                   n_fft=2048, hop_length=512, feature_type='mel', 
                   n_mels=128, n_cqt_bins=84):
        """
        Transcribe audio file to MIDI
        
        Args:
            audio_file: Path to audio file
            output_midi_file: Path to output MIDI file
            sample_rate: Sample rate (16000 or 22050 recommended)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length
            feature_type: Type of features to extract ('mel', 'cqt', or 'both')
            n_mels: Number of mel bands
            n_cqt_bins: Number of CQT bins
            
        Returns:
            midi_data: PrettyMIDI object
        """
        # Preprocess audio
        features, audio_length = preprocess_audio(
            audio_file, 
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            feature_type=feature_type,
            n_mels=n_mels,
            n_cqt_bins=n_cqt_bins
        )
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32).T.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(features)
        
        # Convert predictions to numpy
        predictions = predictions.squeeze(0).cpu().numpy()
        
        # Create MIDI file
        midi_data = create_midi_from_predictions(
            predictions, 
            output_file=output_midi_file,
            hop_time=hop_length / sample_rate  # Convert hop length to time
        )
        
        return midi_data 
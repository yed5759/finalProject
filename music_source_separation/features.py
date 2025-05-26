# features.py

import numpy as np
import librosa
import pickle
from pathlib import Path

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

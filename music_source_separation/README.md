# Piano Transformer

A transformer-based system for converting piano WAV audio to MIDI.

## Architecture

This project has a simple and straightforward structure:

- `piano_transformer.py`: Contains all the model components, audio processing, and MIDI utilities
- `main.py`: Command-line interface for using the system
- `prepare_maestro_dataset.py`: Script for preparing the MAESTRO dataset for training
- `train_model.py`: Script for training the piano transformer model
- `evaluate_model.py`: Script for evaluating model performance

## Audio Features

The system supports multiple audio representations:

- **Mel Spectrogram**: Traditional mel-frequency spectrogram
- **Constant-Q Transform (CQT)**: Frequency representation that better matches musical scales
- **Onset Detection**: Automatic detection of note onsets

It also supports explicit modeling of:
- Note onsets (beginnings)
- Note offsets (endings)
- Note velocities (loudness)

## Dataset Preparation

This project is designed to work with the MAESTRO dataset. You can prepare the dataset with:

```bash
# Prepare MAESTRO dataset with default settings (16kHz sample rate)
python prepare_maestro_dataset.py --maestro-dir path/to/maestro --output-dir data

# Prepare with 22.05kHz sample rate
python prepare_maestro_dataset.py --maestro-dir path/to/maestro --output-dir data --sample-rate 22050
```

The script will:
1. Convert audio files to the specified sample rate
2. Organize files into train/validation/test splits based on MAESTRO metadata
3. Create the required directory structure for training

## Training

You can train the model using the MAESTRO dataset:

```bash
# Train with default settings (mel spectrogram features)
python train_model.py --data-dir data --output-dir models

# Train with CQT features
python train_model.py --data-dir data --feature-type cqt --output-dir models/cqt_model

# Train with both mel and CQT features
python train_model.py --data-dir data --feature-type both --sample-rate 22050 --output-dir models/combined_model
```

The training script includes:
- Automatic logging of metrics
- Model checkpointing
- Learning rate scheduling
- Visualization of training progress

## Evaluation

You can evaluate a trained model on the test set:

```bash
# Evaluate model with default settings
python evaluate_model.py --model-path models/best_model.pth --data-dir data

# Evaluate with different feature types
python evaluate_model.py --model-path models/cqt_model/best_model.pth --feature-type cqt --data-dir data
```

The evaluation script measures:
- Precision, Recall, and F1 score
- Note-level accuracy
- Onset and offset detection performance

## Usage

You can use the system for transcription through the command-line interface:

```bash
# Process a single file with default settings (mel spectrogram)
python main.py --input sample.wav --output output.mid

# Process using CQT representation
python main.py --input sample.wav --output output.mid --feature-type cqt

# Process using both mel and CQT features
python main.py --input sample.wav --output output.mid --feature-type both --sample-rate 22050

# Process a directory of files
python main.py --input audio_folder/ --output midi_folder/ --batch

# Use a specific trained model
python main.py --input sample.wav --model models/best_model.pth
```

## Command Line Options

### Main Transcription Script

- `--input, -i`: Input WAV file or directory of WAV files (required)
- `--output, -o`: Output directory or file (default: "output")
- `--model, -m`: Path to trained model (optional)
- `--batch, -b`: Process directory of WAV files (flag)
- `--cpu`: Force CPU processing (flag)
- `--sample-rate, -sr`: Audio sample rate, either 16000 or 22050 Hz (default: 16000)
- `--feature-type, -ft`: Audio feature type: 'mel', 'cqt', or 'both' (default: 'mel')
- `--fft-size, -fs`: FFT window size (default: 2048)

### Dataset Preparation Script

- `--maestro-dir`: Path to MAESTRO dataset (required)
- `--output-dir`: Path to output directory (default: "data")
- `--sample-rate`: Target sample rate, 16000 or 22050 Hz (default: 16000)

### Training Script

- `--data-dir`: Path to data directory (default: "data")
- `--output-dir`: Path to output directory (default: "models")
- `--feature-type`: Audio feature type (default: "mel")
- `--sample-rate`: Audio sample rate (default: 16000)
- `--fft-size`: FFT window size (default: 2048)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 50)

## Requirements

This project requires the following Python packages:

- torch
- librosa
- pretty_midi
- numpy
- soundfile
- tqdm
- matplotlib
- pandas
- mir_eval

You can install them with pip:

```bash
pip install -r requirements.txt
```

## Project Structure

```
piano_transformer/
├── main.py                  # Command-line interface for transcription
├── piano_transformer.py     # All-in-one model file
├── prepare_maestro_dataset.py # Dataset preparation script
├── train_model.py           # Training script
├── evaluate_model.py        # Evaluation script
├── requirements.txt         # Project dependencies
├── README.md                # This README
└── data/                    # Directory for data files
    ├── train/               # Training data
    │   ├── audio/           # Training audio files
    │   └── midi/            # Training MIDI files
    ├── val/                 # Validation data
    │   ├── audio/           # Validation audio files
    │   └── midi/            # Validation MIDI files
    └── test/                # Test data
        ├── audio/           # Test audio files
        └── midi/            # Test MIDI files
``` 
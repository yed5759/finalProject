# Piano Transformer

A transformer-based system for converting piano WAV audio to MIDI.

## Architecture

This project has a simple and straightforward structure:

- `piano_transformer.py`: Contains all the model components, audio processing, and MIDI utilities
- `main.py`: Command-line interface for using the system
- `prepare_maestro_dataset.py`: Script for preparing the MAESTRO dataset for training
- `train_model.py`: Script for training the piano transformer model
- `evaluate_model.py`: Script for evaluating model performance

## Audio Features and Representation

The system uses Constant-Q Transform (CQT) as the audio representation, which better matches the logarithmic nature of musical pitch scales. The model is designed specifically for piano audio:

- **Constant-Q Transform (CQT)**: Frequency representation with logarithmically spaced frequency bins that better match musical scales
- **Explicit Onset-Offset-Velocity Representation**: The model predicts three aspects for each of the 88 piano keys:
  - Note onsets (beginnings)
  - Note offsets (endings) 
  - Note velocities (loudness)

This representation enables more precise transcription with accurate timing and dynamics.

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
# Train with default settings
python train_model.py --data-dir data --output-dir models

# Train with custom CQT bins and sample rate
python train_model.py --data-dir data --cqt-bins 96 --sample-rate 22050 --output-dir models/high_res_model
```

The training script includes:
- Training with explicit onset-offset-velocity targets
- Automatic logging of metrics for each aspect (onset accuracy, offset accuracy, velocity error)
- Model checkpointing
- Learning rate scheduling
- Visualization of training progress

## Evaluation

You can evaluate a trained model on the test set:

```bash
# Evaluate model with default settings
python evaluate_model.py --model-path models/best_model.pth --data-dir data

# Evaluate with different CQT settings
python evaluate_model.py --model-path models/high_res_model/best_model.pth --cqt-bins 96 --sample-rate 22050 --data-dir data
```

The evaluation script measures:
- Onset precision, recall, and F1 score
- Offset precision, recall, and F1 score
- Velocity error
- Overall note-level accuracy

## Usage

You can use the system for transcription through the command-line interface:

```bash
# Process a single file with default settings
python main.py --input sample.wav --output output.mid

# Process with custom CQT settings
python main.py --input sample.wav --output output.mid --cqt-bins 96 --sample-rate 22050

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
- `--fft-size, -fs`: FFT window size (default: 2048)
- `--cqt-bins, -cb`: Number of CQT bins (default: 84)

### Dataset Preparation Script

- `--maestro-dir`: Path to MAESTRO dataset (required)
- `--output-dir`: Path to output directory (default: "data")
- `--sample-rate`: Target sample rate, 16000 or 22050 Hz (default: 16000)

### Training Script

- `--data-dir`: Path to data directory (default: "data")
- `--output-dir`: Path to output directory (default: "models")
- `--sample-rate`: Audio sample rate (default: 16000)
- `--fft-size`: FFT window size (default: 2048)
- `--cqt-bins`: Number of CQT bins (default: 84)
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
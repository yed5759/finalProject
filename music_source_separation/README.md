# Piano Transformer

A deep learning model for automatic piano transcription (converting WAV audio to MIDI) using Constant-Q Transform (CQT) features and a Transformer architecture. The model predicts onset, offset, and velocity values for each of the 88 piano keys.

## Project Structure

```
piano_transformer/
├── dataset/                # Raw dataset files
│   └── MAESTRO/           # MAESTRO dataset 
│       ├── 2004/          # Year-specific folders containing audio and MIDI files
│       ├── 2006/
│       └── ...
├── data/                   # Processed data for training/validation/testing
│   ├── train/             # Training split
│   │   ├── audio/         # Audio files (WAV)
│   │   └── midi/          # MIDI files
│   ├── val/               # Validation split
│   │   ├── audio/
│   │   └── midi/
│   └── test/              # Test split
│       ├── audio/
│       └── midi/
├── models/                 # Saved models
│   └── piano_transformer/ # Model checkpoints and training history
├── output/                 # Generated transcriptions and visualizations
├── piano_transformer.py    # Core model implementation
├── main.py                 # Command-line interface for transcription
├── prepare_maestro_dataset.py # Dataset preparation script
├── train_model.py          # Training script
├── evaluate_model.py       # Evaluation script
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd piano_transformer
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Using the MAESTRO Dataset

This project uses the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (MIDI and Audio Edited for Synchronous TRacks and Organization) for training and evaluation. The dataset should be downloaded and placed in the `dataset/MAESTRO` directory.

You can download the MAESTRO dataset v3.0.0 from: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)

## Data Preparation

To prepare the MAESTRO dataset for training, run:

```bash
python prepare_maestro_dataset.py --maestro-dir dataset/MAESTRO --output-dir data --sample-rate 16000
```

This will:
- Split the dataset into train/val/test sets according to the official split
- Resample audio files to 16kHz (or other specified rate)
- Organize files into the appropriate directories

## Training the Model

To train a new model:

```bash
python train_model.py --data-dir data --model-dir models/piano_transformer --n-cqt-bins 88 --hidden-dim 256 --num-layers 6 --batch-size 16 --epochs 50
```

Key training parameters:
- `--n-cqt-bins`: Number of CQT bins (default: 88, matching piano keys)
- `--hidden-dim`: Hidden dimension of the model (default: 256)
- `--num-layers`: Number of transformer layers (default: 6)
- `--num-heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)
- `--sample-rate`: Audio sample rate (default: 16000)
- `--lr`: Learning rate (default: 0.001)

Training progress and model checkpoints will be saved to the specified `model-dir`.

## Evaluation

To evaluate a trained model on the test set:

```bash
python evaluate_model.py --model-path models/piano_transformer/best_model.pt --data-dir data --output-dir output/evaluation --visualize
```

This will:
- Load the trained model
- Evaluate it on all test files
- Calculate metrics (onset F1, offset F1, velocity RMSE)
- Generate visualizations if the `--visualize` flag is provided

## Transcription

To transcribe a new piano audio file:

```bash
python main.py --audio-file path/to/your/audio.wav --model-path models/piano_transformer/best_model.pt --output-dir output --save-piano-roll
```

This will:
- Load the trained model
- Process the audio file
- Generate a MIDI file with the transcription
- Save a piano roll visualization if the `--save-piano-roll` flag is provided

## Model Architecture

The Piano Transformer uses:
- Constant-Q Transform (CQT) features from audio
- Transformer encoder architecture
- Three output heads for predicting:
  - Note onsets (when a key is pressed)
  - Note offsets (when a key is released)
  - Note velocities (how hard a key is pressed)

## Requirements

- Python 3.7+
- PyTorch 1.9+
- librosa
- pretty_midi
- numpy
- matplotlib
- tqdm
- sklearn

See `requirements.txt` for detailed dependencies.

## Performance

When trained on the MAESTRO dataset, the model achieves:
- Onset F1 score: ~0.80-0.85
- Offset F1 score: ~0.60-0.70
- Velocity RMSE: ~0.10-0.15

Performance varies depending on training time and hyperparameters. 
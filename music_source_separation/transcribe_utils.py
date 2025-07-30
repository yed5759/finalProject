from main import transcribe_audio
import argparse

def transcribe_piano_audio(input_audio, model_path='models/piano_transformer/model.pt', output_dir="output", save_roll=False):
    args = argparse.Namespace(
        audio_file=input_audio,
        model_path=model_path,
        output_dir=output_dir,
        sample_rate=16000,
        hop_length=512,
        n_cqt_bins=88,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        threshold=0.925,
        save_piano_roll=save_roll,
        cpu=False
    )
    return transcribe_audio(args)

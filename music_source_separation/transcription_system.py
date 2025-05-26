# transcription_system.py

import torch
from PianoTransformer import PianoTransformer
from features import process_audio_file
from midi_utils import create_midi_from_predictions


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

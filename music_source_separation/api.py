from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
from audio_features import process_audio_file
from piano_transformer import PianoTransformer
import soundfile as sf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model configuration
MODEL_PATH = 'models/piano_transformer/model.pt'
SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_CQT_BINS = 88
THRESHOLD = 0.5

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PianoTransformer(
    n_cqt_bins=N_CQT_BINS,
    hidden_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1
).to(device)

# Load trained weights
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

model.eval()

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        file.save(temp_file.name)
        
        try:
            # Process the audio file
            features = process_audio_file(
                temp_file.name,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                n_cqt_bins=N_CQT_BINS
            )
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            # Get model predictions
            with torch.no_grad():
                predictions = model(features_tensor)
                # Convert logits to probabilities
                probabilities = torch.sigmoid(predictions[0]).cpu().numpy()
                # Apply threshold
                binary_predictions = (probabilities > THRESHOLD).astype(int)
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            # Return both probabilities and binary predictions
            return jsonify({
                'success': True,
                'probabilities': probabilities.tolist(),
                'predictions': binary_predictions.tolist(),
                'shape': {
                    'time_steps': probabilities.shape[0],
                    'notes': probabilities.shape[1]
                }
            })
            
        except Exception as e:
            # Clean up the temporary file in case of error
            os.unlink(temp_file.name)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
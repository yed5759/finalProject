#!/usr/bin/env python3

import torch
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path
from piano_transformer import PianoTransformer
from audio_features import process_audio_file

def compute_metrics(pred, target):
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    tn = ((pred == 0) & (target == 0)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

def plot_piano_rolls(pred_roll, gt_roll, start=0, end=200):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(gt_roll[start:end].T, aspect='auto', origin='lower', cmap='Greys')
        plt.title('Ground Truth Piano Roll')
        plt.ylabel('MIDI Key')
        plt.subplot(2, 1, 2)
        plt.imshow(pred_roll[start:end].T, aspect='auto', origin='lower', cmap='Greys')
        plt.title('Predicted Piano Roll')
        plt.ylabel('MIDI Key')
        plt.xlabel('Frame')
        plt.tight_layout()
        plt.show()

def simple_evaluate():
    """
    Super simple evaluation: Test your model on ONE audio file and see how it did
    """
    print("🎹 SIMPLE PIANO TRANSCRIPTION EVALUATION")
    print("="*50)
    
    # Step 1: Load your trained model
    print("\n1️⃣ Loading your trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find your model file
    # model_files = list(Path('models/piano_transformer').glob('*.pt'))
    model_files = list(Path('models/checkpoints').glob('model_epoch_10.pt'))
    if not model_files:
        model_files = list(Path('.').glob('*.pt'))
    
    if not model_files:
        print("❌ No model found! Train your model first.")
        return
    
    model_path = model_files[0]  # Use the first model found
    print(f"Using model: {model_path}")
    
    # Load the model (simple version)
    model = PianoTransformer().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # Step 2: Find a test audio file
    print("\n2️⃣ Finding a test audio file...")
    test_files = list(Path('data/test/audio').glob('*.wav'))
    if not test_files:
        print("❌ No test files found! Put some .wav files in data/test/audio/")
        return
    
    audio_file = test_files[0]  # Use the first test file
    print(f"Testing on: {audio_file.name}")
    
    # Step 3: Extract features from the audio
    print("\n3️⃣ Extracting features from audio...")
    features = process_audio_file(audio_file)
    print(f"Audio length: {len(features)} frames ({len(features)*512/16000:.1f} seconds)")
    
    # Step 4: Run your model to get predictions
    print("\n4️⃣ Running your model...")
    
    # Handle long sequences by breaking them into chunks
    max_length = 1000  # Maximum length your model can handle
    
    if len(features) <= max_length:
        # Short audio - process all at once
        print("Processing entire audio at once...")
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_predictions = model(features_tensor)
            predictions = torch.sigmoid(raw_predictions)
        
        predictions = predictions[0].cpu().numpy()
    
    else:
        # Long audio - process in chunks
        print(f"Audio is long ({len(features)} frames), processing in chunks...")
        predictions = np.zeros((len(features), 88))
        
        # Process in overlapping chunks
        chunk_size = max_length
        overlap = 200  # Small overlap between chunks
        
        for start in range(0, len(features), chunk_size - overlap):
            end = min(start + chunk_size, len(features))
            chunk = features[start:end]
            
            print(f"  Processing chunk {start}-{end} ({len(chunk)} frames)")
            
            # Run model on this chunk
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                chunk_pred = model(chunk_tensor)
                chunk_pred = torch.sigmoid(chunk_pred)
            
            chunk_pred = chunk_pred[0].cpu().numpy()
            
            # Store predictions (simple - just overwrite overlaps)
            predictions[start:end] = chunk_pred
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
    
    # Step 5: Load the correct answer (ground truth)
    print("\n5️⃣ Loading the correct answer...")
    midi_file = Path('data/test/midi') / f"{audio_file.stem}.mid"
    
    if not midi_file.exists():
        print(f"❌ No MIDI file found: {midi_file}")
        print("⚠️  Can't compare accuracy, but model is working!")
        show_predictions(predictions)
        return
    
    # Convert MIDI to the same format as predictions
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    piano_roll = midi_data.get_piano_roll(fs=16000/512)  # Same timing as features
    piano_roll = piano_roll[21:109]  # Piano keys only (88 keys)
    piano_roll = piano_roll.T  # Make it [time, keys]
    
    # Make sure both have same length
    min_length = min(len(predictions), len(piano_roll))
    predictions = predictions[:min_length]
    piano_roll = piano_roll[:min_length]
    
    # Convert to binary (0 or 1) - "Is this note playing?"
    ground_truth = (piano_roll > 0).astype(float)
    predicted_notes = (predictions > 0.7).astype(float)  # 0.7 = threshold
    
    print(f"Comparison length: {min_length} frames")
    
    # Step 6: Compare and calculate how good your model is
    print("\n6️⃣ Checking how well your model did...")

    # Sweep over several thresholds
    for thresh in [0.3, 0.5, 0.7, 0.9, 0.925, 0.95]:
        predicted_notes = (predictions > thresh).astype(float)
        correct_predictions = (predicted_notes == ground_truth)
        accuracy = correct_predictions.mean()
        true_positives = ((predicted_notes == 1) & (ground_truth == 1)).sum()
        false_positives = ((predicted_notes == 1) & (ground_truth == 0)).sum()
        false_negatives = ((predicted_notes == 0) & (ground_truth == 1)).sum()
        metrics = compute_metrics(predicted_notes, ground_truth)
        print(f"\n--- Threshold: {thresh:.2f} ---")
        print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"   Precision: {metrics['precision']*100:.1f}%")
        print(f"   Recall: {metrics['recall']*100:.1f}%")
        print(f"   F1 Score: {metrics['f1']*100:.1f}%")
        print(f"   Correct Notes Found: {true_positives}/{int(ground_truth.sum())} ({100*true_positives/max(ground_truth.sum(),1):.1f}%)")
        print(f"   Wrong Notes Added: {false_positives}")
        print(f"   Notes Missed: {false_negatives}")

    # Use default threshold for verdict and visualization
    predicted_notes = (predictions > 0.95).astype(float) # Was 0.5
    correct_predictions = (predicted_notes == ground_truth)
    accuracy = correct_predictions.mean()
    true_positives = ((predicted_notes == 1) & (ground_truth == 1)).sum()
    false_positives = ((predicted_notes == 1) & (ground_truth == 0)).sum()
    false_negatives = ((predicted_notes == 0) & (ground_truth == 1)).sum()

    # Plot the first 200 frames (about 6 seconds at 512 hop, 16kHz)
    plot_piano_rolls(predicted_notes, ground_truth, start=0, end=200)

    # Step 7: Give a simple verdict
    print(f"\n🎯 VERDICT:")
    if accuracy > 0.95:
        print("🟢 AMAZING! Your model is working excellently!")
    elif accuracy > 0.90:
        print("🟢 GREAT! Your model is working very well!")
    elif accuracy > 0.85:
        print("🟡 GOOD! Your model is working well!")
    elif accuracy > 0.75:
        print("🟠 OKAY! Your model is learning but could be better!")
    else:
        print("🔴 NEEDS WORK! Your model needs more training!")
    
    # Step 8: Show some examples
    show_predictions(predictions, ground_truth)

def show_predictions(predictions, ground_truth=None):
    """Show what notes the model thinks are playing"""
    print(f"\n🎵 WHAT YOUR MODEL HEARS:")
    print("-" * 30)
    
    # Look at first 10 seconds (about 312 frames at 16kHz/512)
    sample_frames = min(312, len(predictions))
    
    # Count how many notes are active in each frame
    for i in range(0, sample_frames, 31):  # Every ~1 second
        frame = predictions[i]
        active_notes = (frame > 0.5).sum()
        max_confidence = frame.max()
        
        second = i * 512 / 16000  # Convert frame to seconds
        print(f"At {second:.1f}s: {active_notes} notes playing (max confidence: {max_confidence:.2f})")
        
        # Show which specific notes if not too many
        if active_notes <= 5 and active_notes > 0:
            note_indices = np.where(frame > 0.5)[0]
            note_names = [f"Key{idx}" for idx in note_indices]
            print(f"         Notes: {', '.join(note_names)}")
    
    if ground_truth is not None:
        print(f"\n📈 SUMMARY:")
        avg_notes_true = ground_truth.mean(axis=0).sum()
        avg_notes_pred = predictions.mean(axis=0).sum()
        print(f"   Average notes playing (true): {avg_notes_true:.1f}")
        print(f"   Average notes playing (predicted): {avg_notes_pred:.1f}")

if __name__ == "__main__":
    simple_evaluate()
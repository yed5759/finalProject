#!/usr/bin/env python3

import torch
import numpy as np
import pretty_midi
from pathlib import Path
from piano_transformer import PianoTransformer, process_audio_file

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
    model_files = list(Path('models/piano_transformer').glob('*.pt'))
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
    predicted_notes = (predictions > 0.5).astype(float)  # 0.5 = threshold
    
    print(f"Comparison length: {min_length} frames")
    
    # Step 6: Compare and calculate how good your model is
    print("\n6️⃣ Checking how well your model did...")
    
    # Count correct predictions
    correct_predictions = (predicted_notes == ground_truth)
    accuracy = correct_predictions.mean()
    
    # Count active notes
    total_true_notes = ground_truth.sum()
    total_predicted_notes = predicted_notes.sum()
    
    # Count hits and misses
    true_positives = ((predicted_notes == 1) & (ground_truth == 1)).sum()
    false_positives = ((predicted_notes == 1) & (ground_truth == 0)).sum()
    false_negatives = ((predicted_notes == 0) & (ground_truth == 1)).sum()
    
    print(f"📊 RESULTS:")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Correct Notes Found: {true_positives}/{total_true_notes} ({100*true_positives/max(total_true_notes,1):.1f}%)")
    print(f"   Wrong Notes Added: {false_positives}")
    print(f"   Notes Missed: {false_negatives}")
    
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
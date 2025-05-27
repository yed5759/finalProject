# midi_utils.py

import pretty_midi
import numpy as np
import torch

# ===== MIDI Utilities =====

def create_midi_from_predictions(note_presence, output_file, threshold=0.5, tempo=120.0, velocity=80):
    """
    Convert model predictions to a MIDI file using note presence and velocity representation
    
    Args:
        predictions: Model predictions, can be:
                    - A single array with shape (time_steps, pitch_bins*3)
                    - A tuple of (note_presence, velocity) each with shape (time_steps, 88)
        output_file: Output MIDI file path
        threshold: Threshold for detecting note onsets
        tempo: Tempo of the output MIDI file in BPM
    """
    # Create MIDI file
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Convert to probabilities if needed, using sigmoid
    if np.max(np.abs(note_presence)) > 1.0:
        note_presence = 1.0 / (1.0 + np.exp(-note_presence))

    # Process each time step
    for t in range(len(note_presence)):
        # Find active notes
        active_notes = np.where(note_presence[t] > threshold)[0]
        
        for note in active_notes:
            # Create MIDI note
            vel = int(velocity[t, note] * 127) if isinstance(velocity, (np.ndarray, torch.Tensor)) else velocity
            midi_note = pretty_midi.Note(
                velocity=vel,  # Scale to MIDI velocity range
                pitch=note + 21,  # Convert to MIDI pitch (A0 = 21)
                start=t * 0.01,  # 10ms per frame
                end=(t + 1) * 0.01
            )
            piano.notes.append(midi_note)
    
    # Add piano to MIDI file and save
    midi_data.instruments.append(piano)
    midi_data.write(output_file)
    
    return midi_data

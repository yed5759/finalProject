import pretty_midi
import numpy as np

def piano_roll_to_note_tuples(piano_roll, hop_length, sample_rate):
    """
    Converts a binary piano roll (time_steps x 88) into a list of note dicts:
    { pitch, startTime, duration }

    pitch: MIDI pitch (21–108)
    startTime/duration: in seconds
    """
    note_tuples = []
    seconds_per_frame = hop_length / sample_rate
    time_steps, num_pitches = piano_roll.shape

    for pitch in range(num_pitches):
        active = piano_roll[:, pitch]
        in_note = False
        for t in range(time_steps):
            if active[t] and not in_note:
                start_time = t * seconds_per_frame
                in_note = True
                start_idx = t
            elif not active[t] and in_note:
                end_time = t * seconds_per_frame
                duration = end_time - start_time
                note_tuples.append({
                    "pitch": pitch + 21,
                    "startTime": round(start_time, 5),
                    "duration": round(duration, 5)
                })
                in_note = False
        # Handle notes that sustain till the end
        if in_note:
            end_time = time_steps * seconds_per_frame
            duration = end_time - start_time
            note_tuples.append({
                "pitch": pitch + 21,
                "startTime": round(start_time, 5),
                "duration": round(duration, 5)
            })

    return note_tuples


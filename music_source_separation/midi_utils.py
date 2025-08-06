import pretty_midi

def midi_to_vexflow_note(midi_number):
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    pitch_class = midi_number % 12
    octave = (midi_number // 12) - 1
    note_name = NOTE_NAMES[pitch_class]
    return f"{note_name}/{octave}"

def duration_to_vexflow(duration_sec, tempo_bpm=120):
    beat_duration = 60 / tempo_bpm
    beats = duration_sec / beat_duration

    # Map to standard note durations
    if abs(beats - 4) < 0.3:
        return "w"  # whole
    elif abs(beats - 2) < 0.25:
        return "h"  # half
    elif abs(beats - 1) < 0.2:
        return "q"  # quarter
    elif abs(beats - 0.5) < 0.1:
        return "8"  # eighth
    elif abs(beats - 0.25) < 0.05:
        return "16"  # sixteenth
    else:
        return "q"  # fallback for unrecognized durations

def note_tuples_to_vexflow(note_tuples, tempo_bpm=120):
    vex_notes = []
    sorted_notes = sorted(note_tuples, key=lambda x: x["startTime"])

    for note in sorted_notes:
        vex_note = {
            "keys": [midi_to_vexflow_note(note["pitch"])],
            "duration": duration_to_vexflow(note["duration"], tempo_bpm)
        }
        vex_notes.append(vex_note)
    return vex_notes

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


import librosa
import numpy as np

def midi_to_vexflow_key(midi_number: int) -> str:
    name = librosa.midi_to_note(midi_number)
    letter, octave = name[:-1], name[-1]
    return f"{letter.lower()}/{octave}"

def seconds_to_duration(seconds: float, bpm: int = 120) -> str:
    quarter_len = 60.0 / bpm
    beats = seconds / quarter_len

    if isinstance(beats, np.ndarray):
        if beats.size == 1:
            beats = beats.item()
        else:
            raise ValueError(f"Expected scalar, got array with shape {beats.shape}")

    beats_rounded = round(beats, 3)

    duration_map = {
        4: "w",      # whole
        2: "h",      # half
        1: "q",      # quarter
        0.5: "8",    # eighth
        0.25: "16",  # sixteenth
        0.125: "32",
        0.0625: "64",
    }
    if beats_rounded in duration_map:
        return duration_map[beats_rounded]

        # dotted notes (1.5 * base)
    for base, symbol in duration_map.items():
        if abs(beats_rounded - 1.5 * base) < 0.1:
            return symbol + "d"
    return "q"  # fallback
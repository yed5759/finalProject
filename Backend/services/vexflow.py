import librosa
import numpy as np


def midi_to_vexflow_key(midi_number: int) -> str:
    name = librosa.midi_to_note(midi_number)
    letter, octave = name[:-1], name[-1]
    letter = (
        letter.replace("♯", "#")
        .replace("♭", "b")
        .lower()
    )

    return f"{letter.lower()}/{octave}"

def seconds_to_duration(seconds: float, bpm: int = 120):
    quarter_len = 60.0 / bpm
    beats = seconds / quarter_len

    if isinstance(beats, np.ndarray):
        if beats.size == 1:
            beats = beats.item()
        else:
            raise ValueError(f"Expected scalar, got array with shape {beats.shape}")

    beats_rounded = round(beats, 3)

    duration_map = {
        4: "w", 2: "h", 1: "q",
        0.5: "8", 0.25: "16", 0.125: "32", 0.0625: "64",
    }

    if beats_rounded in duration_map:
        return {"duration": duration_map[beats_rounded], "triplet": False}

    # dotted
    for base, symbol in duration_map.items():
        if abs(beats_rounded - 1.5 * base) < 0.1:
            return {"duration": symbol + "d", "triplet": False}

    # triplet
    for base, symbol in duration_map.items():
        if abs(beats_rounded - (2/3) * base) < 0.1:
            return {"duration": symbol, "triplet": True}

    return {"duration": "q", "triplet": False}

from collections import defaultdict

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from services.vexflow import midi_to_vexflow_key, seconds_to_duration
from utils.inputProcessors import download_audio
from onsets_and_frames import *
import torch
import soundfile as sf
import numpy as np
import librosa
import os
import uuid
import resampy

home_routes = Blueprint("home", __name__)
CHORD_TOLERANCE = 0.03  # 30ms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(
    BASE_DIR,
    "..", "static",
    "model-500000.pt"
))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

@home_routes.route("/home", methods=["POST"])
def create_notes():
    content = request.form.get('url')
    if content:
        filepath, title = download_audio(content)
    else:
        content = request.files.get('file')
        title = secure_filename(content.filename)
        base_dir = os.path.dirname(os.path.dirname(__file__))  # עולה תיקייה מעל Backend/routes
        save_dir = os.path.join(base_dir, "temp")
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, title)
        content.save(filepath)

    audio, sr = sf.read(filepath, dtype='float32')

    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)

    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE, filter="kaiser_fast")
        sr = SAMPLE_RATE

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    mel_spectrogram = melspectrogram(audio_tensor)
    mel_spectrogram = mel_spectrogram.transpose(-1, -2)

    with torch.no_grad():
        onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel_spectrogram)

    pitches, intervals, velocities = extract_notes(
        onset_pred[0], frame_pred[0], velocity_pred[0]
    )

    y, sr = librosa.load(filepath)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    midi_pitches = pitches + MIN_MIDI
    scaling = HOP_LENGTH / SAMPLE_RATE
    times = intervals * scaling

    notes = []
    for pitch, (start, end), vel in zip(midi_pitches, times, velocities):
        start = float(start)
        end = float(end)
        vel = float(vel)

        notes.append({
            "pitch": int(pitch),
            "start": float(start),
            "end": float(end),
            "velocity": float(vel),
            "duration": float(end - start)
        })

    chords_dict = defaultdict(list)
    for note in notes:
        rounded_start = round(note["start"] / CHORD_TOLERANCE) * CHORD_TOLERANCE
        chords_dict[rounded_start].append(note)

    render_notes = []

    for start_time, group in chords_dict.items():
        if len(group) == 1:
            note = group[0]
            render_notes.append({
                "type": "note",
                "pitches": [note["pitch"]],
                "start": note["start"],
                "duration": note["duration"],
                "velocity": note["velocity"]
            })
        else:
            pitches = [n["pitch"] for n in group]
            start = min(n["start"] for n in group)
            end = max(n["end"] for n in group)
            velocity = np.mean([n["velocity"] for n in group])

            render_notes.append({
                "type": "chord",
                "pitches": pitches,
                "start": start,
                "duration": end - start,
                "velocity": velocity
            })
    vexflow_notes = []
    for note in render_notes:
        vexflow_notes.append({
            "id" : str(uuid.uuid4()),
            "keys" : [midi_to_vexflow_key(p) for p in note["pitches"]],
            "duration": seconds_to_duration(note["duration"], bpm=tempo)
        })
    os.remove(filepath)
    return jsonify({'redirect': f'/Notes?songName={title}',
                    'notes': vexflow_notes}), 200
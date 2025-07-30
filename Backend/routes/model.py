from flask import Blueprint, request, jsonify
from Backend.utils.inputProcessors import download_audio
from music_source_separation.transcribe_utils import transcribe_piano_audio

home_routes = Blueprint("home", __name__)

@home_routes.route("/home", methods=["POST"])
def create_notes():
    content = request.form.get('url')
    if content:
        filepath = download_audio(content)
    else:
        content = request.files.get('file')
        filename = content.filename
        content.save(f'../temp/{filename}')
        filepath = f'../temp/{filename}'

    prediction = transcribe_piano_audio(filepath)
    if not prediction:
        return 'there was an error in the prediction', 422
    return jsonify({'notes': prediction['notes']}), 201
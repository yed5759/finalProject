from flask import Blueprint, request, jsonify
from Backend.utils.inputProcessors import download_audio
from music_source_separation.transcribe_utils import transcribe_piano_audio

home_routes = Blueprint("home", __name__)

@home_routes.route("/home", methods=["POST"])
def create_notes():
    content = request.form.get('url')
    if content:
        filepath, title = download_audio(content)
    else:
        content = request.files.get('file')
        title = content.filename
        content.save(f'../temp/{title}')
        filepath = f'../temp/{title}'

    prediction = transcribe_piano_audio(filepath)
    if not prediction:
        return 'there was an error in the prediction', 422
    return jsonify({'redirect': f'/Notes?songName={title}',
                    'notes': prediction['notes']}), 200
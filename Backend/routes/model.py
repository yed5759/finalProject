from flask import Blueprint, request, jsonify
from Backend.utils.inputProcessors import download_audio

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
    return jsonify({})
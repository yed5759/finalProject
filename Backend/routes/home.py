from flask import Blueprint, request, jsonify
from Backend.utils.inputProcessors import download_audio

home_page = Blueprint('home_page', __name__)

@home_page.route('/home', methods=['POST'])
def upload_handler():
    file = request.files.get('file')
    url = request.form.get('url')

    if file:
        filename = file.filename
        file.save(f"../temp/{filename}")
        return jsonify({"status" : "file uploaded successfully", "filename" : filename}), 200
    elif url:
        file, name = download_audio(url)
        if not file:
            return jsonify({"status" : "file not downloaded"}), 400

        return jsonify({"status" : "file downloaded successfully", "filename": name}), 200
    else:
        return jsonify({"status" : "no file or url provided"}), 400






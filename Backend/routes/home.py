from flask import Blueprint, request, redirect
from Backend.utils.auth import require_auth
from Backend.utils.inputProcessors import download_audio

home_page = Blueprint('home_page', __name__)

@home_page.route('/home', methods=['POST'])
@require_auth
def upload():
    data = request.form.to_dict()
    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename != "":
        print("Got a file:", uploaded_file.filename)
    elif data['url']:
        download_audio(data['url'])
    else:
        return "No valid input provided", 400

    return redirect('/Notes')







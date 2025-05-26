# routes/sharing.py

from flask import Blueprint, request, jsonify
from utils.user import get_user_by_token
from utils.sharing import (
    share_song_to_user,
    get_shared_songs_for_user, 
    accept_shared_song
)

sharing_routes = Blueprint("sharing", __name__)

# Helper to extract user from request header
def get_user_from_request():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Unauthorized"
    token = auth_header.split(" ")[1]
    try:
        user = get_user_by_token(token)
        return user, None
    except Exception as e:
        return None, str(e)

# POST /share
@sharing_routes.route("/share", methods=["POST"])
def share_song():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401

    payload = request.get_json()
    recipient_username = payload.get("username")
    song_data = payload.get("song")

    if not recipient_username or not song_data:
        return jsonify({"error": "Username and song data required"}), 400

    share_song_to_user(user["_id"], recipient_username, song_data)
    return jsonify({"message": "Song shared"})

# GET /shared
@sharing_routes.route("/shared", methods=["GET"])
def get_shared():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401

    shared_songs = get_shared_songs_for_user(user["_id"])
    return jsonify(shared_songs)

# POST /shared/accept/<song_id>
@sharing_routes.route("/shared/accept/<song_id>", methods=["POST"])
def accept_shared(song_id):
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401

    accepted_song = accept_shared_song(user["_id"], song_id)
    return jsonify(accepted_song)

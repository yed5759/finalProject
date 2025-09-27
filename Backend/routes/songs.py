# routes/songs.py

from flask import Blueprint, request, jsonify
from utils.user import get_user_by_token
from utils.songs import (
    get_songs_for_user, add_song_to_user,
    delete_song_from_user, update_song_for_user,
    get_song_by_id
)

songs_routes = Blueprint("songs", __name__)

# Middleware-like helper to extract user from token
def get_user_from_request():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Unauthorized"
    token = auth_header.split(" ")[1]
    try:
        user = get_user_by_token(token)
        return user, None
    except Exception as e:
        print("Token validation error in /songs:", e)
        return None, str(e)

# GET /songs
@songs_routes.route("/songs", methods=["GET"])
def get_songs():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    return jsonify(get_songs_for_user(user["_id"]))

# GET /songs/<song_id>
@songs_routes.route("/songs/<song_id>", methods=["GET"])
def get_song(song_id):
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    return jsonify(get_song_by_id(user["_id"], song_id)), 200

# POST /songs
@songs_routes.route("/songs", methods=["POST"])
def add_song():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    song_data = request.get_json()
    added_song = add_song_to_user(user["_id"], song_data)
    return jsonify(added_song)

# PUT /songs/<song_id>

@songs_routes.route("/songs/<song_id>", methods=["PUT", "PATCH"])
def update_song(song_id):
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    song_data = request.get_json()
    update_song_for_user(user["_id"], song_id, song_data)
    return jsonify({"message": "Song updated"})

# DELETE /songs/<song_id>
@songs_routes.route("/songs/<song_id>", methods=["DELETE"])
def delete_song(song_id):
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    delete_song_from_user(user["_id"], song_id)
    return jsonify({"message": "Song deleted"})

# GET /songs/public/<owner_id>/<song_id>
@songs_routes.route("/songs/public/<owner_id>/<song_id>", methods=["GET"])
def get_song_public(owner_id, song_id):
    try:
        song = get_song_by_id(owner_id, song_id)
        if not song:
            return jsonify({"error": "Song not found"}), 404
        return jsonify(song), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

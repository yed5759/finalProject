# utils/sharing.py

from utils.auth import db
from uuid import uuid4
from utils.songs import add_song_to_user

# Share a song to another user (adds to their 'shared_songs')
def share_song_to_user(sender_id, recipient_username, song_data):
    recipient = db.users.find_one({"username": recipient_username})
    if not recipient:
        raise ValueError("Recipient user not found")

    # Make sure the song matches the song schema
    shared_song = {
        "id": str(uuid4()), 
        "title": song_data.get("title"),
        "artist": song_data.get("artist"),
        "notes": song_data.get("notes", []),
        "tags": song_data.get("tags", [])
    }

    db.users.update_one(
        {"_id": recipient["_id"]},
        {"$push": {"shared_songs": shared_song}}
    )
    return shared_song

# Get all shared songs for a user
def get_shared_songs_for_user(user_id):
    user = db.users.find_one({"_id": user_id}, {"shared_songs": 1})
    if not user:
        raise ValueError("User not found")
    return user.get("shared_songs", [])

# Accept a shared song and move it to user's own songs
def accept_shared_song(user_id, shared_song_id):
    user = db.users.find_one({"_id": user_id})
    if not user:
        raise ValueError("User not found")

    shared_songs = user.get("shared_songs", [])
    song_to_accept = next((s for s in shared_songs if s["id"] == shared_song_id), None)
    if not song_to_accept:
        raise ValueError("Shared song not found")

    # Remove the shared song from shared_songs
    db.users.update_one(
        {"_id": user_id},
        {"$pull": {"shared_songs": {"id": shared_song_id}}}
    )

    # Remove the existing id so a new one is generated
    song_to_accept.pop("id", None)

    # Add the song using the standard add_song_to_user function
    return add_song_to_user(user_id, song_to_accept)
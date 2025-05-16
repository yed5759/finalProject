# utils/songs.py

from pymongo import ReturnDocument
from utils.auth import db
from uuid import uuid4

# Get songs list for a user by their sub (user id)
def get_songs_for_user(user_id):
    user = db.users.find_one({"_id": user_id}, {"songs": 1})
    if not user:
        raise ValueError("User not found")
    return user.get("songs", [])

# Add a new song to user's songs list
def add_song_to_user(user_id, song_data):
    # Ensure required fields
    if "title" not in song_data:
        raise ValueError("Title is required")

    song_data["id"] = str(uuid4())  # Assign a UUID
    song_data.setdefault("artist", None)
    song_data.setdefault("notes", [])
    song_data.setdefault("tags", [])

    result = db.users.find_one_and_update(
        {"_id": user_id},
        {"$push": {"songs": song_data}},
        return_document=ReturnDocument.AFTER
    )
    if not result:
        raise ValueError("User not found")
    return song_data

# Delete a song by its UUID from user's songs
def delete_song_from_user(user_id, song_id):
    result = db.users.update_one(
        {"_id": user_id},
        {"$pull": {"songs": {"id": song_id}}}
    )
    if result.modified_count == 0:
        raise ValueError("Song or user not found")

# Update a song by its UUID within user's songs
def update_song_for_user(user_id, song_id, updated_data):
    update_query = {f"songs.$.{key}": value for key, value in updated_data.items()}
    result = db.users.update_one(
        {"_id": user_id, "songs.id": song_id},
        {"$set": update_query}
    )
    if result.modified_count == 0:
        raise ValueError("Song or user not found")

# Get a specific song by its UUID from user's songs
def get_song_by_id(user_id, song_id):
    user = db.users.find_one({"_id": user_id}, {"songs": 1})
    if not user:
        raise ValueError("User not found")

    songs = user.get("songs", [])
    for song in songs:
        if song.get("id") == song_id:
            return song

    raise ValueError("Song not found")


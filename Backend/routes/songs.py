# routes/songs.py

from fastapi import APIRouter, Depends, HTTPException
from utils.user import get_user_by_token
from utils.songs import (
    get_songs_for_user, add_song_to_user,
    delete_song_from_user, update_song_for_user,
    get_song_by_id
)
from fastapi.security import OAuth2PasswordBearer
from typing import Dict

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.get("/songs")
def get_songs(token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    return get_songs_for_user(user["_id"])

@router.get("/songs/{song_id}")
def get_song(song_id: str, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    return get_song_by_id(user["_id"], song_id)

@router.post("/songs")
def add_song(song_data: Dict, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    return add_song_to_user(user["_id"], song_data)

@router.put("/songs/{song_id}")
def update_song(song_id: str, song_data: Dict, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    update_song_for_user(user["_id"], song_id, song_data)
    return {"message": "Song updated"}

@router.delete("/songs/{song_id}")
def delete_song(song_id: str, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    delete_song_from_user(user["_id"], song_id)
    return {"message": "Song deleted"}

# routes/sharing.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from utils.user import get_user_by_token
from utils.sharing import (
    share_song_to_user,
    get_shared_songs_for_user, accept_shared_song
)
from utils.songs import add_song_to_user
from typing import Dict

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/share")
def share_song(payload: Dict, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    recipient_username = payload.get("username")
    song_data = payload.get("song")
    if not recipient_username or not song_data:
        raise HTTPException(status_code=400, detail="Username and song data required")
    share_song_to_user(user["_id"], recipient_username, song_data)
    return {"message": "Song shared"}

@router.get("/shared")
def get_shared(token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    return get_shared_songs_for_user(user["_id"])

@router.post("/shared/accept/{song_id}")
def accept_shared(song_id: str, token: str = Depends(oauth2_scheme)):
    user = get_user_by_token(token)
    return accept_shared_song(user["_id"], song_id)

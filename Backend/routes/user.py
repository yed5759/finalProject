# routes/user.py

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from utils.user import get_user_by_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.get("/me")
def get_me(token: str = Depends(oauth2_scheme)):
    return get_user_by_token(token)

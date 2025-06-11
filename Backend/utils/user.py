# utils/user.py

import os
import jwt
from dotenv import load_dotenv
from utils.db import get_db

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
 
# Get user from database by id_token
def get_user_by_token(id_token):
    try:
        db = get_db()
        decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        
        sub = decoded.get("sub")
        if not sub:
            raise ValueError("Token missing sub")

        user = db.users.find_one({"_id": sub}) 
        if not user:
            raise ValueError("User not found")

        return user
    except Exception as e:
        raise Exception(f"Failed to get user: {str(e)}")

# utils/user.py

import os
import jwt
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["taking_notes"]
 
# Get user from database by id_token
def get_user_by_token(id_token):
    try:
        decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        sub = decoded.get("sub")
        if not sub:
            raise ValueError("Token missing sub")

        user = db.users.find_one({"_id": sub}, {"_id": 0})  # hide internal MongoDB _id
        if not user:
            raise ValueError("User not found")

        return user
    except Exception as e:
        raise Exception(f"Failed to get user: {str(e)}")

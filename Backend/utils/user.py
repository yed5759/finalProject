# utils/auth.py

import os
import jwt
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["taking_notes"]

# Extract 'sub' from the id_token
def extract_sub_from_token(id_token):
    try:
        decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        return decoded.get("sub")
    except Exception:
        return None

# Ensures user exists in DB; if not creating user in DB
def ensure_user_exists(id_token):
    try:
        decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        sub = decoded.get("sub")

        if not sub:
            raise ValueError("Token missing sub")

        user = db.users.find_one({"_id": sub})
        if not user:
            return create_new_user(id_token)

    except Exception as e:
        raise Exception(f"Failed to ensure user: {str(e)}")


# Create a new user in the MongoDB collection
def create_new_user(id_token):
    decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
    sub = decoded.get("sub")
    display_name = decoded.get("name", "New User")

    if not sub:
        raise ValueError("Token missing sub")

    new_user = {
        "_id": sub,
        "displayName": display_name,
        "songs": []
    }
    db.users.insert_one(new_user)
    
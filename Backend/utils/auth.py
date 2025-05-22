# utils/auth.py

import os
import requests
import jwt
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["taking_notes"]

# Exchange code for tokens
def exchange_code_for_tokens(code):
    # Load env vars
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI")
    cognito_domain = os.getenv("COGNITO_DOMAIN")

    # Exchange code for tokens
    token_url = f"https://{cognito_domain}/oauth2/token"

    payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(
        token_url, 
        data=payload, 
        headers=headers,
        auth=(client_id, client_secret)
    )

    if response.status_code != 200:
        return None, response.text

    return response.json(), None

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

        return user 
    
    except Exception as e:
        raise Exception(f"Failed to ensure user: {str(e)}")

# Create a new user in the MongoDB collection
def create_new_user(id_token):
    decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
    
    import json
    print("Cognito token fields:\n", json.dumps(decoded, indent=4))

    sub = decoded.get("sub")
    username = decoded.get("cognito:username", "New User")

    if not sub:
        raise ValueError("Token missing sub")

    new_user = {
        "_id": sub,
        "username": username,
        "songs": [],
        "shared_songs": []
    }
    db.users.insert_one(new_user)
   
    return new_user
# routes/user.py

from flask import Blueprint, request, jsonify
from utils.user import get_user_by_token
from utils.db import get_db

user_routes = Blueprint("user", __name__)

# Helper to extract user from Authorization header
def get_user_from_request():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Unauthorized"
    token = auth_header.split(" ")[1]
    try:
        user = get_user_by_token(token)
        return user, None
    except Exception as e:
        print("Token validation error:", e)
        return None, str(e)

# GET /me
@user_routes.route("/me", methods=["GET"])
def get_me():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
        db = get_db()
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    users_collection = db["users"]
    mongo_user = users_collection.find_one({"_id": user["sub"]})
    if mongo_user is None:
        # המשתמש לא קיים – נחזיר מבנה ריק עם מידע מה-token בלבד
        return jsonify({
            "sub": user["sub"],
            "username": user.get("username", ""),
            "songs": [],
            "shared_songs": []
        })

    return jsonify({
        "sub": mongo_user["_id"],
        "username": mongo_user.get("username", user.get("username")),
        "songs": mongo_user.get("songs", []),
        "shared_songs": mongo_user.get("shared_songs", [])
    })

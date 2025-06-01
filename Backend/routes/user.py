# routes/user.py

from flask import Blueprint, request, jsonify
from utils.user import get_user_by_token

user_routes = Blueprint("user", __name__)

# Helper to extract user from Authorization header
def get_user_from_request():
    auth_header = request.headers.get("Authorization")
    # todo delete
    print("Authorization header:", auth_header)
  
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Unauthorized"
    token = auth_header.split(" ")[1]
    try:
        user = get_user_by_token(token)
        return user, None
    except Exception as e:
        # todo delete
        print("Token validation error:", e)
        return None, str(e)

# GET /me
@user_routes.route("/me", methods=["GET"])
def get_me():
    user, error = get_user_from_request()
    if error:
        return jsonify({"error": error}), 401
    return jsonify(user)

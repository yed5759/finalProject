# routes/auth.py

from flask import Blueprint, request, jsonify
from utils.auth import (
    exchange_code_for_tokens,
    ensure_user_exists
)
    
auth_routes = Blueprint("auth", __name__)

@auth_routes.route("/auth/callback", methods=["GET"])
def auth_callback():
    code = request.args.get("code")

    if not code:
        return jsonify({"error": "Missing code"}), 400

    tokens, error = exchange_code_for_tokens(code)
    if error:
        return jsonify({"error": "Failed to exchange code", "details": error}), 400

    try:
        id_token = tokens.get("id_token")

        # Ensure the user exists in the DB, but don't return user data
        user = ensure_user_exists(id_token)

        if not user:
            return jsonify({"error": "User creation failed"}), 500 


        return jsonify({
            "id_token": tokens["id_token"],
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token") 
        })

    except Exception as e:
        return jsonify({"error": "Failed to process user", "details": str(e)}), 400

@auth_routes.route("/auth/refresh", methods=["POST"])
def refresh_token():
    payload = request.get_json()
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        return jsonify({"error": "No refresh token"}), 400
    # Implement refresh token validation against Cognito and generate new id_token
    new_tokens = {
        "id_token": "...",
        "access_token": "...",
        "refresh_token": refresh_token  # or new
    }
    return jsonify(new_tokens)

from flask import Blueprint, request, jsonify
from utils.auth import exchange_code_for_tokens

auth_routes = Blueprint("auth", __name__)

# Auth callback route from Cognito with ?code=...
@auth_routes.route("/auth/callback", methods=["GET"])
def auth_callback():
    code = request.args.get("code")

    if not code:
        return jsonify({"error": "Missing code"}), 400

    tokens, error = exchange_code_for_tokens(code)

    if error:
        return jsonify({"error": "Failed to exchange code", "details": error}), 400

    return jsonify({
        "id_token": tokens["id_token"],
        "access_token": tokens["access_token"]
    })

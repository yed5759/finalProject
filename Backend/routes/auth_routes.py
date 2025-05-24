from flask import Blueprint, request, jsonify
from Backend.services.auth_service import exchange_code_for_tokens

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route("/callback", methods=["GET"])
def auth_callback():
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing code"}), 400

    try:
        tokens = exchange_code_for_tokens(code)
        return jsonify({
            "id_token": tokens["id_token"],
            "access_token": tokens["access_token"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
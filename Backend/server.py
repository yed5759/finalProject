from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder="static")

# Static file routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/<path:path>")
def catch_all(path):
    return send_from_directory(app.static_folder, path)

# Auth callback route
@app.route("/auth/callback", methods=["POST"])
def auth_callback():
    data = request.json
    code = data.get("code")

    if not code:
        return jsonify({"error": "Missing code"}), 400

    # todo delete print
    print("📥 Received code from frontend:", code)  # ✅ הדפסת הקוד לטרמינל

    # Load env vars
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI")
    cognito_domain = os.getenv("COGNITO_DOMAIN")

    # Exchange code for tokens
    token_url = f"{cognito_domain}/oauth2/token"
    payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(token_url, data=payload, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Failed to exchange code", "details": response.text}), 400

    tokens = response.json()

    # todo delete print
    print("✅ Tokens received:", tokens)
    return jsonify(tokens)
    # You’ll add the token exchange logic here
    return jsonify({"message": "Received code", "code": code})




if __name__ == "__main__":
    app.run(debug=True)
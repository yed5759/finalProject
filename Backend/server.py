from flask import Flask, request, jsonify, redirect, make_response
from flask import send_from_directory
from dotenv import load_dotenv
import os
import requests
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Enable CORS for the frontend domain
CORS(app, origins = "http://localhost:3000", supports_credentials=True)

# Auth callback route from Cognito with ?code=...
@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    code = request.args.get("code")

    if not code:
        return jsonify({"error": "Missing code"}), 400

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
    print(response.status_code)
    print(response)

    if response.status_code != 200:
        return jsonify({"error": "Failed to exchange code", "details": response.text}), 400

    tokens = response.json()

    return jsonify({"id_token": tokens["id_token"], "access_token": tokens["access_token"]})

if __name__ == "__main__":
    app.run(debug=True)
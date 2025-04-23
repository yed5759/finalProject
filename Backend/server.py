from flask import Flask, request, jsonify, redirect, make_response
from flask import send_from_directory
from dotenv import load_dotenv
import os
import requests
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# app = Flask(__name__, static_folder="static")


# Enable CORS for the frontend domain
CORS(app, origins = "http://localhost:3000", supports_credentials=True)

# Static file routes

# Serve index.html for root
@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve all other static files
@app.route("/<path:path>")
def catch_all(path):
    return send_from_directory(app.static_folder, path)

# Auth callback route from Cognito with ?code=...
@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    code = request.args.get("code")
    # data = request.json
    # code = data.get("code")

    if not code:
        # return "Missing code", 400
        print("Missing code")
        return jsonify({"error": "Missing code"}), 400

    # todo delete print
    print("📥 Received code from client:", code)  # ✅ הדפסת הקוד לטרמינל

    # Load env vars
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI")
    cognito_domain = os.getenv("COGNITO_DOMAIN")

    # Exchange code for tokens
    # token_url = f"{cognito_domain}/oauth2/token"
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
        print("❌ Error response:", response.text)
        print("###################################")
        # return f"Token exchange failed: {response.text}", 400
        return jsonify({"error": "Failed to exchange code", "details": response.text}), 400

    tokens = response.json()
    # todo delete print
    print("✅ Tokens received:", tokens)

    # redirect_uri = f"http://localhost:3000/home?id_token={tokens['id_token']}&access_token={tokens['access_token']}"

    # return redirect(redirect_uri)

    # Create response and set tokens as HttpOnly cookies
    res = make_response(jsonify({"message": "Tokens set in cookies"}))
    # res = make_response(redirect("http://localhost:3000/home"))
    res.set_cookie("id_token", tokens["id_token"], httponly=True, secure=False, path='/', samesite="Lax")
    res.set_cookie("access_token", tokens["access_token"], httponly=True, secure=False, path='/', samesite="Lax")
    # todo decide where and if saving the refresh token on server side or just let the session end and ask for new tokens
    # res.set_cookie("refresh_token", tokens.get("refresh_token"), httponly=True, secure=False)
    return res

    # return redirect("http://localhost:3000/home")
    # return jsonify(tokens)
    # # You’ll add the token exchange logic here
    # return jsonify({"message": "Received code", "code": code})




if __name__ == "__main__":
    app.run(debug=True)
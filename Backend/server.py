from flask import Flask, request, jsonify, send_from_directory, redirect, make_response
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder="static")

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
        return "Missing code", 400
        return jsonify({"error": "Missing code"}), 400

    # todo delete print
    print("📥 Received code from Cognito redirect:", code)  # ✅ הדפסת הקוד לטרמינל

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
        print("❌ Error response:", response.text)
        return f"Token exchange failed: {response.text}", 400
        return jsonify({"error": "Failed to exchange code", "details": response.text}), 400

    tokens = response.json()
    # todo delete print
    print("✅ Tokens received:", tokens)

    # Create response and set tokens as HttpOnly cookies
    res = make_response(redirect("http://localhost:3000/home"))
    res.set_cookie("id_token", tokens["id_token"], httponly=True, secure=False, path='/', samesite="Lax")
    res.set_cookie("access_token", tokens["access_token"], httponly=True, secure=False, path='/', samesite="Lax")
    # todo decide where and if saving the refresh token on server side or just let the session end and ask for new tokens
    # res.set_cookie("refresh_token", tokens.get("refresh_token"), httponly=True, secure=False)
    return res

    return redirect("http://localhost:3000/home")
    return jsonify(tokens)
    # You’ll add the token exchange logic here
    return jsonify({"message": "Received code", "code": code})




if __name__ == "__main__":
    app.run(debug=True)
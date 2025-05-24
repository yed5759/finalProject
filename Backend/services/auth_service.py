# app/services/auth_service.py

import os
import requests

def exchange_code_for_tokens(code: str):
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI")
    cognito_domain = os.getenv("COGNITO_DOMAIN")

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
        raise Exception(f"Token exchange failed: {response.text}")

    return response.json()
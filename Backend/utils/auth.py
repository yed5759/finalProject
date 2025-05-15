# utils/auth.py

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Exchange code for tokens
def exchange_code_for_tokens(code):
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

    if response.status_code != 200:
        return None, response.text

    return response.json(), None

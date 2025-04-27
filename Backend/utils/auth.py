from functools import wraps
from flask import request, jsonify
from jose import jwt
import requests

# Replace these with your actual Cognito values:
COGNITO_REGION = 'us-east-1'
USER_POOL_ID = 'us-east-1_XXXXXX'
CLIENT_ID = 'your_cognito_app_client_id'

# Download JWKS from Cognito (you can cache this later for performance)
JWKS_URL = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json'
JWKS = requests.get(JWKS_URL).json()

def verify_token(token):
    try:
        headers = jwt.get_unverified_headers(token)
        kid = headers['kid']
        key = next(k for k in JWKS['keys'] if k['kid'] == kid)

        payload = jwt.decode(
            token,
            key,
            algorithms=['RS256'],
            audience=CLIENT_ID,
            issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}'
        )
        return payload
    except Exception as e:
        print(f"[AUTH] Token verification failed: {e}")
        return None

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Authorization header missing'}), 401

        user = verify_token(token)
        if not user:
            return jsonify({'error': 'Invalid or expired token'}), 401

        request.user = user  # Attach decoded token payload to request object
        return f(*args, **kwargs)
    return decorated_function
from flask import Flask
from flask_cors import CORS
from routes.auth import auth_routes

app = Flask(__name__)

# Enable CORS for the frontend domain
CORS(app, origins = "http://localhost:3000", supports_credentials=True)

# Register auth routes
app.register_blueprint(auth_routes)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask
from flask_cors import CORS

from routes.auth import auth_routes
from routes.user import user_routes
from routes.songs import songs_routes
from routes.sharing import sharing_routes

app = Flask(__name__)

# Enable CORS for the frontend domain
CORS(app, origins = "http://localhost:3000", supports_credentials=True)

# Register auth routes
app.register_blueprint(auth_routes)
app.register_blueprint(user_routes)
app.register_blueprint(songs_routes)
app.register_blueprint(sharing_routes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

def create_app():
    load_dotenv()  # Load env variables

    app = Flask(__name__)

    # Enable CORS for frontend
    CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

    # Register routes
    from Backend.routes.auth_routes import auth_bp
    from Backend.routes.home import home_page
    app.register_blueprint(auth_bp)
    app.register_blueprint(home_page)

    return app

app = create_app()
app.run(debug=True)
from flask import Flask
from routes.landing_page import landing_page

def create_app():
    app = Flask(__name__, static_folder="static")
    app.register_blueprint(landing_page)
    return app
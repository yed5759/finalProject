# utils/db.py

from pymongo import MongoClient, errors
import os

client = None

def get_db():
    global client
    if client is None:
        try:
            client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
        except errors.ConnectionFailure:
            print("⚠️ MongoDB unavailable.")
            return None
    return client["YourDatabaseName"]

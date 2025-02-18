import json
import os

class Cache:
    def __init__(self, path):
        os.makedirs(path, 700, exist_ok=True)
        self.__metadata = os.path.join(path, "metadata_downloads.txt")
        if not os.path.exists(self.__metadata):
            with open(self.__metadata, "w") as f:
                json.dump({}, f, indent=4)

    def load(self):
        with open(self.__metadata, "r") as f:
            return json.loads(f)

    def save(self, data):
        with open(self.__metadata, "w") as f:
            json.dump(data, f, indent=4)

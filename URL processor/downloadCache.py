import json
import os
import time


class Cache:
    def __init__(self, path):
        os.makedirs(path, 700, exist_ok=True)
        self.__metadata = os.path.join(path, "metadata_downloads.txt")
        self.__timeThreshold = 3
        if not os.path.exists(self.__metadata):
            with open(self.__metadata, "w") as f:
                json.dump({}, f, indent=4)

    def load(self):
        with open(self.__metadata, "r") as f:
            return json.loads(f)

    def save(self, data):
        with open(self.__metadata, "w") as f:
            json.dump(data, f, indent=4)

    def expired(self):
        time_stamp = time.mktime(time.gmtime())
        load = self.load()

        expired = [audio_id for audio_id in list(load)
                   if time_stamp - time.mktime(
                time.strptime(load[audio_id]["last access"], "%a %b %d %H:%M:%S %Y")) >= 3 * 24 * 60 * 60]

        for audio_id in expired:
            os.remove(load[audio_id]['filepath'])
            del load[audio_id]

        self.save(load)

        with open("./cache/downloads.txt", "r+") as f:
            lines = [line for line in f if not any(audio_id in line for audio_id in expired)]
            f.seek(0)
            f.writelines(lines)
            f.truncate()



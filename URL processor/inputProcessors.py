import yt_dlp
import time
import os
from downloadCache import Cache

CACHE_DIR = "./cache"
ARCHIVE_FILE = os.path.join(CACHE_DIR, "downloads.txt")

"""
handle downloading of the audio from youtube if not already downloaded.
give a timestamp for every audio.
"""
def download_audio(url):
    cache = Cache(CACHE_DIR)
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "./cache/%(title)s.%(ext)s",
        "download_archive": ARCHIVE_FILE,
        "quiet": True,
    }
    try:
        data = cache.load()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)

            if not info['id'] in data:
                ydl.download([url])

            data[info['id']] = {"last access": time.asctime(time.gmtime())}
            cache.save(data)

    except Exception as e:
        print(f"Error downloading audio: {e}")

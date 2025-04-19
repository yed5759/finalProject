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
    cache.expired()
    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "./cache/%(title)s.mp3",
        "download_archive": ARCHIVE_FILE,
        "quiet": True,
    }
    try:
        data = cache.load()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_file = f'./cache/{info["title"]}.mp3'
            data[info['id']] = { "filename": info['filepath'],
                                "last access": time.asctime(time.gmtime())}
            cache.save(data)
            return final_file

    except Exception as e:
        print(f"Error downloading audio: {e}")

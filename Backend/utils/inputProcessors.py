import os

import yt_dlp


"""
handle downloading of the audio from youtube if not already downloaded.
give a timestamp for every audio.
"""
def download_audio(url):
    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "./temp/%(title)s.%(ext)s",
        "quiet": True,
    }
    try:
        # download file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            name = info['requested_downloads'][0]['filepath']

        if not os.path.isfile(name):
            raise FileNotFoundError(f"File not found: {name}")
        return name, info['title']

    except Exception as e:
        print(f"Error downloading audio: {e}")




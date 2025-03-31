import os
import re
import time

from dotenv import load_dotenv
import yt_dlp
import boto3

"""
handle downloading of the audio from youtube if not already downloaded.
give a timestamp for every audio.
"""
load_dotenv()
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
table = dynamodb.Table('SongsMetadata')


def download_audio(url):
    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "%(title)s.%(ext)s",
        "quiet": True,
    }
    try:
        # download metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            name = sanitize_filename(f'{info["title"]}.mp3')
        response = table.get_item(Key={'Song Name': name})

        # check if the file already in s3
        if 'Item' in response:
            new_expire_time = int(time.time()) + (10 * 24 * 60 * 60)
            table.update_item(Key={'Song Name': name}, UpdateExpression="SET #expire = :expire_time",
                              ExpressionAttributeNames={'#expire': 'expire time'},
                              ExpressionAttributeValues={':expire_time': new_expire_time},
                              ReturnValues="UPDATED_NEW")
            return name

         # download file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            name = info['requested_downloads'][0]['filepath']

        if not os.path.isfile(name):
            raise FileNotFoundError(f"File not found: {name}")

        with open(name, "rb") as file_data:
            s3.upload_fileobj(file_data, 'songcache', name)

        table.put_item(Item={
            'Song Name': name,
            'Song URL': url,
            'expire time': int(time.time()) + (10 * 24 * 60 * 60)
        })
        os.remove(name)
        return name

    except Exception as e:
        print(f"Error downloading audio: {e}")


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

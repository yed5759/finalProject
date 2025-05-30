import os
import re
import time
import subprocess

from dotenv import load_dotenv
import yt_dlp
import boto3

"""
handle downloading of the audio from youtube if not already downloaded.
give a timestamp for every audio.
"""
subprocess.run([
    "openssl", "enc", "-aes-256-cbc", "-d",
    "-in", ".env.enc", "-out", ".env",
    "-pass", "file:../secret.key"
], check=True)

load_dotenv()
os.remove("../.env.enc")

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
        "outtmpl": "../temp/%(title)s.%(ext)s",
        "quiet": True,
    }
    try:
        # download metadata
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            name = sanitize_filename(f'{info["title"]}.mp3')
        song_key = os.path.basename(name)
        response = table.get_item(Key={'Song Name': song_key})

        # check if the file already in s3
        if 'Item' in response:
            new_expire_time = int(time.time()) + (10 * 24 * 60 * 60)
            table.update_item(Key={'Song Name': song_key}, UpdateExpression="SET #expire = :expire_time",
                              ExpressionAttributeNames={'#expire': 'expire time'},
                              ExpressionAttributeValues={':expire_time': new_expire_time},
                              ReturnValues="UPDATED_NEW")
            s3.copy_object(Bucket="songscache",
                           CopySource={'Bucket': 'songscache', 'Key': f"upload/{song_key}"},
                           Key=f"uploads/{song_key}")
            local_file = f"../Backend/temp/{song_key}"
            s3.download_file("songscache", f"upload/{song_key}", local_file)
            return local_file

        # download file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            name = info['requested_downloads'][0]['filepath']

        if not os.path.isfile(name):
            raise FileNotFoundError(f"File not found: {name}")

        with open(name, "rb") as file_data:
            s3.upload_fileobj(file_data, 'songscache', f"uploads/{os.path.basename(name)}")

        table.put_item(Item={
            'Song Name': os.path.basename(name),
            'Song URL': url,
            'expire time': int(time.time()) + (10 * 24 * 60 * 60)
        })
        return name

    except Exception as e:
        print(f"Error downloading audio: {e}")


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

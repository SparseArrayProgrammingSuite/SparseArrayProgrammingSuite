import hashlib
import os
import hashlib

from .config import config
import requests

def download(url):
    """
    Download a file from the specified URL
    
    Args:
        url (str): The URL of the file to download.
    """
    filename = url.split("/")[-1]
    path = config.data_path() / filename
    if not path.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        with open(path, 'rb') as f:
            hash = hashlib.file_digest(f, 'sha256').hexdigest()
        if hash not in filename:
            raise ValueError(f"Hash mismatch for downloaded file {filename}. Expected hash in filename, got {hash}.")

def upload(path):
    """
    Upload a file and return the URL where it can be accessed.
    
    Args:
        path (str): The path to the file to upload.
    """
    import boto3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('sparse-array-programming-suite')
    filename = os.path.basename(path)
    with open(path, 'rb') as f:
        hash = hashlib.file_digest(f, 'sha256').hexdigest()
    name, ext = os.path.splitext(filename)
    key = f"{filename}-{hash}{ext}"
    obj = bucket.Object(key)
    obj.upload_file(path)
    url=f"https://{bucket.name}.s3.amazonaws.com/{obj.key}"
    print(
        f"Uploaded file {path} into bucket {bucket.name} with key {obj.key}."
        f"File can be accessed at: {url}"
    )
    return url
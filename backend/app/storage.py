import os
import boto3
from botocore.client import Config

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET = os.getenv("MINIO_BUCKET", "sanitized")

s3 = boto3.resource(
    's3',
    endpoint_url=f'http://{MINIO_ENDPOINT}',
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

def ensure_bucket():
    try:
        s3.create_bucket(Bucket=BUCKET)
    except Exception:
        pass

def upload_fileobj(fileobj, key):
    ensure_bucket()
    s3.Bucket(BUCKET).put_object(Key=key, Body=fileobj)
    return f"minio://{BUCKET}/{key}"

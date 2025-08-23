import os, sys, time
from rq import Connection, Worker, Queue
from redis import Redis
from sanitize import blur_faces_and_save, extract_object_crops
from backend.app.storage import upload_fileobj
import boto3

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(redis_url)
listen = ['default']

def sanitize_job(local_path, original_name):
    print("Sanitizing", local_path)
    blurred_bytes = blur_faces_and_save(local_path)
    key = f"sanitized/{os.path.basename(local_path)}.blur.jpg"
    upload_fileobj(blurred_bytes, key)
    # get crops (if any)
    crops = extract_object_crops(local_path)
    for idx, crop in enumerate(crops):
        crop_key = f"sanitized/crops/{os.path.basename(local_path)}_{idx}.jpg"
        upload_fileobj(crop, crop_key)
    # delete original file immediately
    try:
        os.remove(local_path)
    except:
        pass
    return {"status":"done"}

if __name__ == "__main__":
    with Connection(redis_conn):
        worker = Worker(map(Queue, listen))
        worker.work()
#devpush
from .storage import upload_fileobj
from redis import Redis
from rq import Queue

def enqueue_sanitize(local_path, original_name):
    # This function is executed in worker (import path must match)
    # wrapper for worker — not used by FastAPI directly in this file
    return {"path": local_path, "name": original_name}

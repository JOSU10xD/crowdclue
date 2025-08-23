import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from redis import Redis
from rq import Queue
from .storage import upload_fileobj
from .sanitize_job import enqueue_sanitize

app = FastAPI(title="CrowdClue Backend")

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(redis_url)
q = Queue(connection=redis_conn)

ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "admin@example.com").split(",")

# Simple admin dependency (Day0). Replace with SSO + MFA for prod.
def admin_guard(x_api_key: str = ""):
    # placeholder - enforce real auth in prod
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), authorized=Depends(admin_guard)):
    tmp_dir = "/tmp/uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # enqueue sanitize job: sanitize worker will delete original file
    job = q.enqueue(enqueue_sanitize, tmp_path, file.filename)
    return {"status": "enqueued", "job_id": job.get_id()}

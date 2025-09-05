import os
from io import BytesIO
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, abort
import psycopg2
import psycopg2.extras
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- Config -----------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/usr/src/app/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://hack:hackpwd@postgres:5432/mydb")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "letmein")

# ----------------- YOLO model -----------------
# This downloads yolov8n.pt (~6 MB) on first run and caches it
yolo_model = YOLO("yolov8n.pt")

def sanitize_and_save(image_bytes, out_path):
    """
    - Strips EXIF
    - Resizes to max width 1024
    - Detects all persons using YOLOv8 and blurs them fully
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        if pil.width > 1024:
            ratio = 1024 / float(pil.width)
            pil = pil.resize((1024, int(pil.height * ratio)), Image.LANCZOS)
        pil.save(out_path, format="JPEG", quality=80)
        return

    # Resize large images
    h, w = img.shape[:2]
    if w > 1024:
        new_h = int(h * (1024 / float(w)))
        img = cv2.resize(img, (1024, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Run YOLO person detection
    results = yolo_model.predict(img, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    k = max(31, (min(roi.shape[:2]) // 7) | 1)  # blur kernel
                    blurred = cv2.GaussianBlur(roi, (k, k), 0)
                    img[y1:y2, x1:x2] = blurred

    # Save sanitized image
    success, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise RuntimeError("Failed to encode sanitized image")
    with open(out_path, "wb") as fh:
        fh.write(buf.tobytes())

# ----------------- Flask app -----------------
app = Flask(__name__, static_folder=None)

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
              id SERIAL PRIMARY KEY,
              title TEXT,
              description TEXT,
              filename TEXT,
              uploader TEXT,
              uploaded_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS clues (
              id SERIAL PRIMARY KEY,
              image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
              contributor TEXT,
              text TEXT,
              status TEXT DEFAULT 'pending',
              created_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS users (
              id SERIAL PRIMARY KEY,
              name TEXT UNIQUE,
              score INTEGER DEFAULT 0
            );
            """)
            conn.commit()
    app.logger.info("DB initialized")

# -------- Routes --------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/images", methods=["GET"])
def list_images():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, title, description, filename, uploaded_at FROM images ORDER BY uploaded_at DESC LIMIT 100")
            rows = cur.fetchall()
            for r in rows:
                r["url"] = f"/uploads/{r['filename']}" if r["filename"] else None
            return jsonify(rows)

@app.route("/api/images/<int:image_id>", methods=["GET"])
def image_detail(image_id):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, title, description, filename, uploaded_at FROM images WHERE id=%s", (image_id,))
            img = cur.fetchone()
            if not img:
                return jsonify({"error":"not found"}), 404
            cur.execute("SELECT id, contributor, text, status, created_at FROM clues WHERE image_id=%s ORDER BY created_at DESC", (image_id,))
            clues = cur.fetchall()
            img["url"] = f"/uploads/{img['filename']}" if img["filename"] else None
            return jsonify({"image": img, "clues": clues})

@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    safe = pathlib.Path(UPLOAD_DIR) / filename
    if not safe.exists():
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/admin/images", methods=["POST"])
def admin_upload():
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f = request.files["image"]
    fname = f.filename or f"upload-{int(datetime.utcnow().timestamp())}.jpg"
    fname_safe = "".join([c for c in fname if c.isalnum() or c in (" ", ".", "_", "-")]).replace(" ", "_")
    out_name = f"{int(datetime.utcnow().timestamp())}-{fname_safe}"
    out_path = os.path.join(UPLOAD_DIR, out_name)

    content = f.read()
    try:
        sanitize_and_save(content, out_path)
    except Exception as e:
        app.logger.exception("sanitize failed")
        return jsonify({"error": "processing_failed", "detail": str(e)}), 500

    title = request.form.get("title") or fname_safe
    description = request.form.get("description")
    uploader = request.form.get("uploader") or "admin"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO images (title, description, filename, uploader) VALUES (%s,%s,%s,%s) RETURNING id",
                        (title, description, out_name, uploader))
            new_id = cur.fetchone()[0]
            conn.commit()
    return jsonify({"ok": True, "id": new_id, "filename": out_name, "url": f"/uploads/{out_name}"})

@app.route("/api/images/<int:image_id>/clues", methods=["POST"])
def submit_clue(image_id):
    data = request.get_json() or {}
    contributor = data.get("contributor")
    text = data.get("text")
    if not contributor or not text:
        return jsonify({"error":"missing contributor/text"}), 400
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO clues (image_id, contributor, text) VALUES (%s,%s,%s)", (image_id, contributor, text))
            cur.execute("INSERT INTO users (name, score) VALUES (%s, 0) ON CONFLICT (name) DO NOTHING", (contributor,))
            conn.commit()
    return jsonify({"ok": True})

@app.route("/admin/clues/<int:clue_id>/accept", methods=["POST"])
def accept_clue(clue_id):
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, contributor, status FROM clues WHERE id=%s", (clue_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error":"not found"}), 404
            if row["status"] == "accepted":
                return jsonify({"ok": True, "note": "already accepted"}), 200
            contributor = row["contributor"]
            cur.execute("UPDATE clues SET status='accepted' WHERE id=%s", (clue_id,))
            cur.execute("INSERT INTO users (name, score) VALUES (%s, 10) "
                        "ON CONFLICT (name) DO UPDATE SET score = users.score + 10", (contributor,))
            conn.commit()
    return jsonify({"ok": True})

@app.route("/api/scoreboard", methods=["GET"])
def scoreboard():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT name, score FROM users ORDER BY score DESC LIMIT 20")
            return jsonify(cur.fetchall())

# -------- Run --------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)

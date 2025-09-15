# backend/app/main.py
import os
import shutil
from io import BytesIO
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import psycopg2.errors
import pathlib
import cv2
import numpy as np
import bcrypt
import jwt
from functools import wraps

# secret for JWT (dev only; set in env for production)
JWT_SECRET = os.environ.get("JWT_SECRET", "dev_jwt_secret_change_me")
JWT_ALGO = "HS256"
JWT_EXP_SECONDS = 60 * 60 * 24  # 1 day

# ----------------- Config -----------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/usr/src/app/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://hack:hackpwd@postgres:5432/mydb")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "letmein")

# Try to import ultralytics; if missing, we still run but without segmentation
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Load YOLO segmentation model lazily
yolo_model = None
if YOLO is not None:
    try:
        yolo_model = YOLO("yolov8n-seg.pt")
    except Exception as e:
        print("Warning: YOLO model init failed:", e)
        yolo_model = None

# Load Haar face cascade once
_face_cascade = None
def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade

# === helper auth decorators and functions ===
def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def create_jwt(payload: dict) -> str:
    payload = payload.copy()
    payload["exp"] = int(datetime.utcnow().timestamp()) + JWT_EXP_SECONDS
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    # pyjwt may return bytes in some older versions; ensure str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def decode_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

def auth_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            data = decode_jwt(token)
            if data and "name" in data:
                request.user = data  # attach
                return func(*args, **kwargs)
        return jsonify({"error": "unauthorized"}), 401
    return wrapper

def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            data = decode_jwt(token)
            if data and data.get("is_admin"):
                request.user = data
                return func(*args, **kwargs)
        # fallback to ADMIN_SECRET header (existing)
        auth = request.headers.get("x-admin-secret") or request.args.get("secret")
        if auth == ADMIN_SECRET:
            request.user = {"name": "admin", "is_admin": True}
            return func(*args, **kwargs)
        return jsonify({"error": "unauthorized"}), 401
    return wrapper

# Skin detection, blur helpers
def detect_skin_mask_bgr(img_bgr):
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)
    cr_min, cr_max = 135, 180
    cb_min, cb_max = 85, 135
    cr_mask = (Cr >= cr_min) & (Cr <= cr_max)
    cb_mask = (Cb >= cb_min) & (Cb <= cb_max)
    skin_mask = cr_mask & cb_mask
    skin_mask = (skin_mask.astype(np.uint8) * 255)
    k = max(3, (min(img_bgr.shape[:2]) // 400) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return (skin_mask > 0)

def blur_regions_on_mask(img, mask_bool, blur_kernel=(51,51)):
    if mask_bool.sum() == 0:
        return img
    blurred_full = cv2.GaussianBlur(img, blur_kernel, 0)
    out = img.copy()
    for c in range(3):
        ch = out[:, :, c]
        ch[mask_bool] = blurred_full[:, :, c][mask_bool]
        out[:, :, c] = ch
    return out

def _looser_skin_in_mask(img_bgr, mask_bool):
    h, w = img_bgr.shape[:2]
    skin_loose = detect_skin_mask_bgr(img_bgr)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = img_ycrcb[:, :, 1]
    Cb = img_ycrcb[:, :, 2]
    cr_min, cr_max = 130, 190
    cb_min, cb_max = 75, 140
    looser = (Cr >= cr_min) & (Cr <= cr_max) & (Cb >= cb_min) & (Cb <= cb_max)
    combined = (skin_loose | looser) & mask_bool
    combined_uint = (combined.astype(np.uint8) * 255)
    k = max(3, (min(h, w) // 400) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(combined_uint, cv2.MORPH_OPEN, kernel)
    return (cleaned > 0)

def sanitize_and_save(image_bytes, out_path, debug=False):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        if pil.width > 1024:
            ratio = 1024 / float(pil.width)
            pil = pil.resize((1024, int(pil.height * ratio)), Image.LANCZOS)
        pil.save(out_path, format="JPEG", quality=80)
        return
    h, w = img.shape[:2]
    if w > 1024:
        new_h = int(h * (1024 / float(w)))
        img = cv2.resize(img, (1024, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    skin_mask_global = detect_skin_mask_bgr(img)
    final_blur_mask = np.zeros((h, w), dtype=bool)

    yolo_results = []
    if yolo_model is not None:
        try:
            yolo_results = yolo_model.predict(img, imgsz=640, verbose=False)
        except Exception:
            yolo_results = []

    person_regions = []
    for r in yolo_results:
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            try:
                mask_iter = masks.data
            except Exception:
                mask_iter = masks
            cls_list = []
            if boxes is not None and getattr(boxes, "cls", None) is not None:
                try:
                    cls_list = [int(x) for x in boxes.cls]
                except Exception:
                    cls_list = []
            for i, mask in enumerate(mask_iter):
                cls_id = cls_list[i] if i < len(cls_list) else None
                label = None
                if cls_id is not None and hasattr(yolo_model, "names"):
                    try:
                        label = yolo_model.names[int(cls_id)]
                    except Exception:
                        label = None
                if label is not None and label != "person":
                    continue
                try:
                    mask_np = mask.cpu().numpy()
                except Exception:
                    mask_np = np.asarray(mask)
                if mask_np.shape != (h, w):
                    mask_resized = cv2.resize((mask_np * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask_resized > 127).astype(np.uint8).astype(bool)
                else:
                    mask_bin = (mask_np > 0.5).astype(bool)
                person_regions.append({"mask": mask_bin})
        elif boxes is not None and getattr(boxes, "cls", None) is not None:
            try:
                for i, cls_t in enumerate(boxes.cls):
                    try:
                        cls_id = int(cls_t)
                    except Exception:
                        continue
                    label = None
                    if hasattr(yolo_model, "names"):
                        try:
                            label = yolo_model.names[cls_id]
                        except Exception:
                            label = None
                    if label != "person":
                        continue
                    try:
                        coords = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords[:4])
                    except Exception:
                        try:
                            coords = boxes.xyxy[i]
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        except Exception:
                            continue
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    person_regions.append({"box": (x1, y1, x2, y2)})
            except Exception:
                pass

    face_cascade = _get_face_cascade()
    face_boxes = []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (fx, fy, fw, fh) in faces:
            fx0, fy0, fx1, fy1 = fx, fy, fx + fw, fy + fh
            fx0, fy0 = max(0, fx0), max(0, fy0)
            fx1, fy1 = min(w, fx1), min(h, fy1)
            face_boxes.append((fx0, fy0, fx1, fy1))
    except Exception:
        face_boxes = []

    for region in person_regions:
        if "mask" in region:
            pmask = region["mask"].astype(bool)
            skin_inside = pmask & skin_mask_global
            face_inside_mask = np.zeros_like(final_blur_mask)
            for (fx0, fy0, fx1, fy1) in face_boxes:
                face_box_mask = np.zeros_like(final_blur_mask)
                face_box_mask[fy0:fy1, fx0:fx1] = True
                if np.any(face_box_mask & pmask):
                    face_inside_mask |= face_box_mask
            combined = (skin_inside | face_inside_mask)
            if combined.sum() < 50:
                looser_skin = _looser_skin_in_mask(img, pmask)
                combined |= looser_skin
            if combined.sum() < 50 and np.any(face_inside_mask):
                combined |= face_inside_mask
            final_blur_mask |= combined
        else:
            x1, y1, x2, y2 = region["box"]
            skin_inside = skin_mask_global[y1:y2, x1:x2]
            local_mask = np.zeros((h, w), dtype=bool)
            local_mask[y1:y2, x1:x2] = skin_inside
            face_inside_mask = np.zeros_like(final_blur_mask)
            for (fx0, fy0, fx1, fy1) in face_boxes:
                face_box_mask = np.zeros_like(final_blur_mask)
                face_box_mask[fy0:fy1, fx0:fx1] = True
                if np.any(face_box_mask[y1:y2, x1:x2]):
                    face_inside_mask |= face_box_mask
            combined = local_mask | face_inside_mask
            if combined.sum() < 50 and np.any(face_inside_mask):
                combined |= face_inside_mask
            final_blur_mask |= combined

    if len(person_regions) == 0 and len(face_boxes) > 0:
        for (fx0, fy0, fx1, fy1) in face_boxes:
            final_blur_mask[fy0:fy1, fx0:fx1] = True

    if final_blur_mask.sum() > 0:
        k = max(7, (min(h, w) // 400) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        final_blur_mask = cv2.dilate(final_blur_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        img = blur_regions_on_mask(img, final_blur_mask, blur_kernel=(51,51))

    success, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise RuntimeError("Failed to encode sanitized image")
    with open(out_path, "wb") as fh:
        fh.write(buf.tobytes())

# ----------------- Flask app -----------------
app = Flask(__name__, static_folder=None)

# CORS - allow dev origins (for judge/demo). For production replace with specific hosts.
CORS(app, resources={r"/*": {"origins": "*"}},
     expose_headers=["Content-Type", "Authorization", "x-admin-secret"],
     supports_credentials=True)

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """
    Create tables if missing and apply lightweight idempotent schema changes:
      - ensure password_hash column on users
      - ensure unique constraint on (image_id, contributor)
    """
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
            # add password_hash column idempotently
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;")
            # add unique constraint for clues (image_id, contributor)
            cur.execute("""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                JOIN pg_class t ON c.conrelid = t.oid
                WHERE c.conname = 'uq_clues_image_contributor'
              ) THEN
                ALTER TABLE clues
                  ADD CONSTRAINT uq_clues_image_contributor UNIQUE (image_id, contributor);
              END IF;
            EXCEPTION WHEN duplicate_object THEN
              -- ignore
            END;
            $$;
            """)
            conn.commit()
    app.logger.info("DB initialized")

# -------- Routes --------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/hello")
def hello():
    return jsonify({"msg": "Hello from backend (python)"})

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

# -------- Auth endpoints (register/login) --------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    name = data.get("name")
    password = data.get("password")
    if not name or not password:
        return jsonify({"error": "missing name/password"}), 400

    password_hash = _hash_password(password)
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users (name, password_hash, score) VALUES (%s, %s, 0)",
                    (name, password_hash)
                )
                conn.commit()
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return jsonify({"error": "user exists"}), 400
            except Exception as e:
                conn.rollback()
                app.logger.exception("register failed")
                return jsonify({"error": "db_error", "detail": str(e)}), 500
    return jsonify({"ok": True})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    name = data.get("name")
    password = data.get("password")
    if not name or not password:
        return jsonify({"error": "missing name/password"}), 400

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT name, password_hash FROM users WHERE name=%s", (name,))
            row = cur.fetchone()
            if not row or not row.get("password_hash"):
                return jsonify({"error": "invalid credentials"}), 401
            if not _check_password(password, row["password_hash"]):
                return jsonify({"error": "invalid credentials"}), 401

    token = create_jwt({"name": name})
    return jsonify({"ok": True, "token": token})

@app.route("/api/me", methods=["GET"])
@auth_required
def me():
    return jsonify({"user": request.user})

# -------- Upload / clue endpoints --------
@app.route("/admin/images", methods=["POST"])
def admin_upload():
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400

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
        return jsonify({"error":"processing_failed", "detail": str(e)}), 500

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
@auth_required
def submit_clue(image_id):
    data = request.get_json() or {}
    contributor = request.user.get("name")
    text = data.get("text")
    if not text:
        return jsonify({"error":"missing text"}), 400

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (name, score) VALUES (%s, 0) ON CONFLICT (name) DO NOTHING", (contributor,))
            try:
                cur.execute(
                    "INSERT INTO clues (image_id, contributor, text) VALUES (%s,%s,%s) RETURNING id",
                    (image_id, contributor, text)
                )
                clue_id = cur.fetchone()[0]
                conn.commit()
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return jsonify({"error":"already_submitted", "detail":"user already has a clue for this image"}), 409
            except Exception as e:
                conn.rollback()
                app.logger.exception("insert clue failed")
                return jsonify({"error":"db_error","detail":str(e)}), 500
    return jsonify({"ok": True, "clue_id": clue_id})

# ---------- Admin endpoints ----------
@app.route("/admin/clues", methods=["GET"])
def admin_list_clues():
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401

    status = request.args.get("status", "pending")
    q = "SELECT id, image_id, contributor, text, status, created_at FROM clues"
    params = ()
    if status != "all":
        q += " WHERE status=%s"
        params = (status,)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(q + " ORDER BY created_at DESC LIMIT 500", params)
            rows = cur.fetchall()
            return jsonify(rows)

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
            cur.execute(
                "INSERT INTO users (name, score) VALUES (%s, 10) "
                "ON CONFLICT (name) DO UPDATE SET score = users.score + 10",
                (contributor,)
            )
            conn.commit()
    return jsonify({"ok": True})

@app.route("/admin/clues/<int:clue_id>", methods=["DELETE"])
def delete_clue(clue_id):
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT contributor, status FROM clues WHERE id=%s", (clue_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error":"not found"}), 404
            contributor, status = row[0], row[1]
            cur.execute("DELETE FROM clues WHERE id=%s", (clue_id,))
            if status == "accepted":
                cur.execute("UPDATE users SET score = GREATEST(0, score - 10) WHERE name=%s", (contributor,))
            conn.commit()
    return jsonify({"ok": True})

@app.route("/admin/images/<int:image_id>", methods=["DELETE"])
def delete_image(image_id):
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT filename FROM images WHERE id=%s", (image_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error":"not found"}), 404
            filename = row["filename"]
            file_path = os.path.join(UPLOAD_DIR, filename) if filename else None
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                app.logger.exception("failed to remove file: %s", e)
            cur.execute("DELETE FROM images WHERE id=%s", (image_id,))
            conn.commit()
    return jsonify({"ok": True, "deleted_file": bool(filename)})

@app.route("/admin/uploads/<path:filename>", methods=["DELETE"])
def admin_delete_upload_file(filename):
    auth = request.headers.get("x-admin-secret") or request.args.get("secret")
    if auth != ADMIN_SECRET:
        return jsonify({"error":"unauthorized"}), 401

    safe = pathlib.Path(UPLOAD_DIR) / filename
    if not safe.exists():
        return jsonify({"error":"not_found"}), 404
    try:
        safe.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        app.logger.exception("failed to delete upload")
        return jsonify({"error":"delete_failed","detail":str(e)}), 500

@app.route("/api/scoreboard", methods=["GET"])
def scoreboard():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT name, score FROM users ORDER BY score DESC LIMIT 20")
            rows = cur.fetchall()
            return jsonify(rows)

# -------- Run --------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)

import os
import cv2
from PIL import Image
from ultralytics import YOLO  # optional - will throw if not installed
from io import BytesIO
from backend.app.storage import upload_fileobj  # ensure import path matches your package layout

# For Day0 use Haar cascade face detector (lightweight)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def blur_faces_and_save(input_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Expand mask area by a margin to be conservative
    for (x, y, w, h) in faces:
        margin = int(0.4 * max(w, h))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img.shape[1], x + w + margin)
        y1 = min(img.shape[0], y + h + margin)
        roi = img[y0:y1, x0:x1]
        blurred = cv2.GaussianBlur(roi, (51,51), 0)
        img[y0:y1, x0:x1] = blurred
    # Save blurred image to bytes
    _, buf = cv2.imencode('.jpg', img)
    return BytesIO(buf.tobytes())

def extract_object_crops(input_path, confidence=0.3):
    crops = []
    try:
        model = YOLO("yolov8n.pt")  # small model for Day0 - ensure file present or let ultralytics download
        results = model(input_path)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                if conf < confidence:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                img = cv2.imread(input_path)
                crop = img[y1:y2, x1:x2]
                _, buf = cv2.imencode('.jpg', crop)
                crops.append(BytesIO(buf.tobytes()))
    except Exception as e:
        print("YOLO not available or failed:", e)
    return crops

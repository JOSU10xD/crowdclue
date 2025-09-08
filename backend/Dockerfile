FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies for OpenCV + psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Force CPU-only PyTorch + other deps
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so it’s cached inside the image
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L -o /root/.cache/torch/hub/checkpoints/yolov8n.pt \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Copy backend source
COPY app ./app

# Working dir inside app
WORKDIR /usr/src/app/app

# Create uploads directory
RUN mkdir -p /usr/src/app/uploads
VOLUME ["/usr/src/app/uploads"]

# Env vars
ENV PORT=3000
EXPOSE 3000

# Run app
CMD ["python", "main.py"]

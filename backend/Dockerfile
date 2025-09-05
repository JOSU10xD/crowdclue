FROM python:3.11-slim

WORKDIR /usr/src/app

# Install system dependencies for OpenCV + psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app ./app

WORKDIR /usr/src/app/app

# Create uploads directory
RUN mkdir -p /usr/src/app/uploads
VOLUME [ "/usr/src/app/uploads" ]

ENV PORT=3000
EXPOSE 3000

CMD ["python", "main.py"]

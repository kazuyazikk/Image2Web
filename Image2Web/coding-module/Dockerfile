FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and required assets
COPY app.py main.py visualization.py html_generator.py css_generator.py ./
COPY wireframe_detection_model.keras ./wireframe_detection_model.keras
COPY resources/ ./resources/
COPY generated_files/ ./generated_files/
COPY output/ ./output/
COPY sample_output.json ./sample_output.json

# Env vars
ENV MODEL_PATH=/app/wireframe_detection_model.keras
ENV BUCKET_NAME=your-project-id.appspot.com
ENV PYTHONUNBUFFERED=1

# Cloud Run requires listening on $PORT
EXPOSE 8080

# Start Flask app with Gunicorn (no preload)
CMD exec gunicorn --bind :$PORT --workers 1 --timeout 300 app:app

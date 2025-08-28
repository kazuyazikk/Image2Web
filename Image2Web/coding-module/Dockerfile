FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and required assets
COPY app.py main.py visualization.py html_generator.py css_generator.py ./
COPY wireframe_detection_model.keras ./wireframe_detection_model.keras
COPY resources/ ./resources/
COPY generated_files/ ./generated_files/
COPY output/ ./output/
COPY sample_output.json ./sample_output.json

# Create necessary directories
RUN mkdir -p generated_files output

# Set environment variables
ENV MODEL_PATH=/app/wireframe_detection_model.keras
ENV BUCKET_NAME=your-project-id.appspot.com
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Cloud Run requires listening on $PORT
EXPOSE 8080

# Use proper Cloud Run configuration for Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --preload app:app
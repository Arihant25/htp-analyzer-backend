# Backend Dockerfile for HTP Analyzer
# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Configure pip with better timeout and retry settings for slow/unstable connections
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
ENV PIP_NO_CACHE_DIR=1

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only version first (no CUDA dependencies, much smaller and faster)
# This avoids downloading 500+ MB CUDA packages
RUN pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN grep -v -E '^\s*(torch|torchvision|#|$)' requirements.txt > requirements-filtered.txt && \
    pip install -r requirements-filtered.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary application files
COPY app.py main.py mapping.json ./
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p static data/processed data/raw RAG results/training/training_outputs results/evaluation

# Copy only the latest, best-performing model weights into the container
# This ensures the container has the correct model without unnecessary files
COPY results/training/training_outputs/htp_yolo11s_20251029_230000/weights/best.pt ./best.pt

# Expose port
EXPOSE 8000

# Simple health check without requests dependency
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import http.client; h = http.client.HTTPConnection('localhost:8000'); h.request('GET', '/health'); r = h.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

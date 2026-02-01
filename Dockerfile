# Smart AI Energy Predictor - Docker Image
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="SDG7 Energy Predictor"
LABEL description="AI-powered energy consumption prediction and optimization"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY models/ ./models/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p mlruns logs

# Expose ports
# 8501: Streamlit dashboard
# 8000: FastAPI (if enabled)
# 5000: MLflow UI (if enabled)
EXPOSE 8501 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

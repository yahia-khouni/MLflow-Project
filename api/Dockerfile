# ============================================================
# FastAPI Dockerfile
# ============================================================
# Multi-stage-ish Dockerfile for the prediction API.
# Uses Python 3.11 slim to keep the image small.
#
# Web dev analogy: this is like a Dockerfile for a Node.js
# Express app — install deps, copy code, start the server.
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system deps needed by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
# This means if only your code changes (not requirements.txt),
# Docker reuses the cached layer and skips pip install → faster builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start the server
# --host 0.0.0.0  → listen on all interfaces (required for Docker)
# --port 8000     → match the EXPOSE above
# --reload        → auto-restart on code changes (dev convenience)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

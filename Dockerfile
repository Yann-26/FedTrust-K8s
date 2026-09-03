# FedTrust-K8s: Dockerfile for Reproducible Federated Learning Experiments
# Image: fedtrust-k8s:latest (CPU-only)

# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for datasets and results
RUN mkdir -p datasets results/figures

# Set entrypoint
ENTRYPOINT ["python"]

# Default command (can be overridden)
CMD ["--help"]
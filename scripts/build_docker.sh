#!/bin/bash
# Build Docker image for FedTrust-K8s

set -e

echo "=================================================="
echo "Building FedTrust-K8s Docker Image"
echo "=================================================="

# Build the image
docker build -t fedtrust-k8s:latest .

echo ""
echo "✅ Docker build complete!"
echo ""
echo "To run an experiment:"
echo "  docker run --rm -v $(pwd)/datasets:/app/datasets -v $(pwd)/results:/app/results fedtrust-k8s:latest experiments/00_baseline.py"
echo ""
echo "To run with GPU (if available):"
echo "  docker run --rm --gpus all -v $(pwd)/datasets:/app/datasets -v $(pwd)/results:/app/results fedtrust-k8s:latest experiments/00_baseline.py"
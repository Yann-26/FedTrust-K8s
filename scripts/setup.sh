#!/bin/bash
# One-command setup for FedTrust-K8s

set -e

echo "=================================================="
echo "FedTrust-K8s: One-Command Setup"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  Docker Compose not found. Using 'docker compose' instead."
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Build the Docker image
echo ""
echo "📦 Building Docker image..."
$COMPOSE_CMD build

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p datasets results notebooks

# Download MNIST dataset (inside container)
echo ""
echo "📊 Downloading MNIST dataset..."
docker run --rm -v $(pwd)/datasets:/app/datasets fedtrust-k8s:latest -c "
from src.data import get_mnist_datasets
get_mnist_datasets()
print('Dataset downloaded successfully!')
"

# Run a test experiment
echo ""
echo "🧪 Running test experiment..."
docker run --rm \
    -v $(pwd)/datasets:/app/datasets \
    -v $(pwd)/results:/app/results \
    fedtrust-k8s:latest \
    experiments/00_baseline.py

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To run an experiment:"
echo "  docker run --rm -v \$(pwd)/datasets:/app/datasets -v \$(pwd)/results:/app/results fedtrust-k8s:latest experiments/00_baseline.py"
echo ""
echo "Or use Docker Compose:"
echo "  $COMPOSE_CMD run --rm fedtrust experiments/00_baseline.py"
echo ""
echo "To run Jupyter Lab:"
echo "  $COMPOSE_CMD up jupyter"
echo "  (Then open http://localhost:8888 in your browser)"

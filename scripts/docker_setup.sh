#!/usr/bin/env bash
#
# Docker Setup Script for SourceSleuth
# 
# This script initializes Docker volumes and builds the container
# for local MCP server deployment.
#
# Usage:
#   ./scripts/docker_setup.sh              # Linux/macOS
#   bash scripts/docker_setup.sh           # Alternative
#
# For Windows PowerShell, use: scripts\docker_setup.ps1
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "SourceSleuth Docker Setup"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ Docker found: $(docker --version)"
echo ""

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker daemon is running"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Create necessary directories if they don't exist
echo "Creating directories..."
mkdir -p "$PROJECT_ROOT/student_pdfs"
mkdir -p "$PROJECT_ROOT/data"
echo "✓ Directories created"
echo ""

# Build the Docker image
echo "Building Docker image (this may take a few minutes on first run)..."
docker build -t sourcesleuth:latest "$PROJECT_ROOT"

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
    echo ""
else
    echo "✗ Docker build failed"
    exit 1
fi

# Create named volume for persistent data
echo "Creating Docker volume for vector store..."
docker volume create sourcesleuth_vector_data 2>/dev/null || true
echo "✓ Volume created (or already exists)"
echo ""

# Test the container
echo "Testing container..."
docker run --rm \
    -v "$PROJECT_ROOT/student_pdfs:/app/student_pdfs" \
    -v sourcesleuth_vector_data:/app/data \
    sourcesleuth:latest \
    python -c "print('Container test successful')"

if [ $? -eq 0 ]; then
    echo "✓ Container test passed"
    echo ""
else
    echo "✗ Container test failed"
    exit 1
fi

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Add your PDF files to: $PROJECT_ROOT/student_pdfs/"
echo ""
echo "2. Configure your MCP Host (e.g., Claude Desktop) with:"
echo ""
echo '   {'
echo '     "mcpServers": {'
echo '       "sourcesleuth": {'
echo '         "command": "docker",'
echo '         "args": ['
echo '           "run", "-i", "--rm",'
echo '           "-v", "/absolute/path/to/student_pdfs:/app/student_pdfs",'
echo '           "-v", "sourcesleuth_vector_data:/app/data",'
echo '           "sourcesleuth:latest"'
echo '         ]'
echo '       }'
echo '     }'
echo '   }'
echo ""
echo "3. Or run manually for testing:"
echo "   docker run -i --rm \\"
echo "     -v ./student_pdfs:/app/student_pdfs \\"
echo "     -v sourcesleuth_vector_data:/app/data \\"
echo "     sourcesleuth:latest"
echo ""
echo "========================================="

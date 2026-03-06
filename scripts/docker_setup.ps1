# Docker Setup Script for SourceSleuth (PowerShell)
# 
# This script initializes Docker volumes and builds the container
# for local MCP server deployment.
#
# Usage:
#   .\scripts\docker_setup.ps1
#
# For Linux/macOS, use: scripts/docker_setup.sh
#

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "SourceSleuth Docker Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed." -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if Docker daemon is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker daemon is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Get project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Create necessary directories
Write-Host "Creating directories..."
$studentPdfPath = Join-Path $ProjectRoot "student_pdfs"
$dataPath = Join-Path $ProjectRoot "data"

if (!(Test-Path $studentPdfPath)) {
    New-Item -ItemType Directory -Path $studentPdfPath | Out-Null
}
if (!(Test-Path $dataPath)) {
    New-Item -ItemType Directory -Path $dataPath | Out-Null
}
Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

# Navigate to project root
Set-Location $ProjectRoot

# Build the Docker image
Write-Host "Building Docker image (this may take a few minutes on first run)..."
docker build -t sourcesleuth:latest $ProjectRoot

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker image built successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create named volume for persistent data
Write-Host "Creating Docker volume for vector store..."
docker volume create sourcesleuth_vector_data 2>$null
Write-Host "✓ Volume created (or already exists)" -ForegroundColor Green
Write-Host ""

# Test the container
Write-Host "Testing container..."
docker run --rm `
    -v "${studentPdfPath}:/app/student_pdfs" `
    -v sourcesleuth_vector_data:/app/data `
    sourcesleuth:latest `
    python -c "print('Container test successful')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Container test passed" -ForegroundColor Green
} else {
    Write-Host "✗ Container test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "1. Add your PDF files to: $studentPdfPath"
Write-Host ""
Write-Host "2. Configure your MCP Host (e.g., Claude Desktop) with:"
Write-Host ""
Write-Host '   {' -ForegroundColor DarkGray
Write-Host '     "mcpServers": {' -ForegroundColor DarkGray
Write-Host '       "sourcesleuth": {' -ForegroundColor DarkGray
Write-Host '         "command": "docker",' -ForegroundColor DarkGray
Write-Host '         "args": [' -ForegroundColor DarkGray
Write-Host '           "run", "-i", "--rm",' -ForegroundColor DarkGray
Write-Host '           "-v", "C:\path\to\student_pdfs:/app/student_pdfs",' -ForegroundColor DarkGray
Write-Host '           "-v", "sourcesleuth_vector_data:/app/data",' -ForegroundColor DarkGray
Write-Host '           "sourcesleuth:latest"' -ForegroundColor DarkGray
Write-Host '         ]' -ForegroundColor DarkGray
Write-Host '       }' -ForegroundColor DarkGray
Write-Host '     }' -ForegroundColor DarkGray
Write-Host '   }' -ForegroundColor DarkGray
Write-Host ""
Write-Host "3. Or run manually for testing:"
Write-Host "   docker run -i --rm `"
Write-Host "     -v ./student_pdfs:/app/student_pdfs `"
Write-Host "     -v sourcesleuth_vector_data:/app/data `"
Write-Host "     sourcesleuth:latest"
Write-Host ""
Write-Host "========================================="

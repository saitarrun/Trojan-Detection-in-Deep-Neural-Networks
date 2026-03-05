#!/bin/bash

# Nautilus Disk Cleanup Script
# This script removes redundant source files, logs, and temporary artifacts to free up space.

echo "=== Nautilus Disk Cleanup Task ==="

# 1. Remove Redis source code (installed via apt-get in the image)
if [ -d "redis-stable" ]; then
    echo "Removing Redis source directory..."
    rm -rf redis-stable
    rm -f redis-stable.tar.gz
fi

# 2. Clear PyTorch/NVIDIA caches
echo "Cleaning Python/Cuda caches..."
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch

# 3. Truncate large log files
if [ -f "training.log" ]; then
    echo "Truncating training.log..."
    > training.log
fi

# 4. Remove temporary uploads
if [ -d "uploads" ]; then
    echo "Cleaning temporary uploads..."
    rm -rf uploads/*
fi

# 5. Optional: Remove __pycache__
echo "Removing __pycache__ folders..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "=== Cleanup Complete ==="
df -h .

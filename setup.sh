#!/bin/bash

# AKCB Setup Script
# This script sets up the conda environment for the AKCB project

set -e  # Exit on error

echo "=========================================="
echo "Setting up AKCB Environment"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Create conda environment from environment.yml
echo ""
echo "Creating conda environment 'akcb'..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "Environment created successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate akcb"
echo ""
echo "To install the AKCB package in development mode:"
echo "    pip install -e ."
echo ""
echo "To verify the installation:"
echo "    python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""

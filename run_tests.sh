#!/bin/bash

# AKCB Test Runner Script
# This script runs all tests for the AKCB framework

set -e

echo "=========================================="
echo "Running AKCB Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest is not installed. Installing..."
    pip install pytest pytest-cov
fi

# Run unit tests
echo "Running unit tests..."
pytest tests/unit/ -v --cov=akcb --cov-report=term-missing

echo ""
echo "Running integration tests..."
pytest tests/integration/ -v

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

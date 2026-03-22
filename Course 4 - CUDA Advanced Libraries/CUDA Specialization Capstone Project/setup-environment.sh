#!/bin/bash

set -e

# Add pip scripts to PATH so uv commands are available
export PATH="$HOME/.local/bin:$PATH"

echo "========================================="
echo " Setting up UV Project"
echo "========================================="

# Install uv
echo "[1/2] Installing uv..."
pip3 install uv

# Sync dependencies including dev group
echo "[2/2] Syncing dependencies..."
uv sync --group dev --python-preference only-system

echo ""
echo "========================================="
echo " UV Project setup complete!"
echo " Python version: $(uv run python3 --version)"
echo "========================================="
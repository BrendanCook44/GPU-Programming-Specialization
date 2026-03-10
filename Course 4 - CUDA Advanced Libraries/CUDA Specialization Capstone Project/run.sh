#!/bin/bash

set -e

# Add pip scripts to PATH so uv commands are available
export PATH="$HOME/.local/bin:$PATH"

# Persist PATH to ~/.bashrc so uv is available in all terminal sessions
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
source ~/.bashrc

echo "========================================="
echo " Setting up UV Project"
echo "========================================="

# Install uv
echo "[1/3] Installing uv..."
pip3 install uv

# Initialize uv project only if .venv doesn't exist
echo "[2/3] Initializing uv project..."
if [ ! -d ".venv" ]; then
    uv init --no-workspace --python 3.8 --python-preference only-system
else
    echo "  .venv already exists, skipping init..."
fi

# Sync dependencies including dev group
echo "[3/3] Syncing dependencies..."
uv sync --group dev --python-preference only-system

echo ""
echo "========================================="
echo " UV Project setup complete!"
echo " Python version: $(uv run python3 --version)"
echo "========================================="
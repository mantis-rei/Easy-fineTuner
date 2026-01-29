#!/bin/bash
# ============================================================
# VPS SETUP SCRIPT - Run this first on a fresh VPS
# ============================================================
# Usage: bash setup.sh
# ============================================================

set -e

echo "========================================"
echo "  VPS FINE-TUNING SETUP"
echo "========================================"

# Update system (ignore non-critical errors)
echo "Updating system..."
apt-get update -qq 2>/dev/null || echo "Warning: Some repos failed to update, continuing..."
apt-get install -y -qq python3-pip python3-venv git wget curl 2>/dev/null || true

# Create virtual environment
echo "Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 || pip install -q torch

# Install training dependencies
echo "Installing training libraries..."
pip install -q transformers datasets peft trl bitsandbytes accelerate sentencepiece protobuf

# Verify installation
echo ""
echo "Checking installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')" 2>/dev/null || echo "GPU check skipped"

echo ""
echo "========================================"
echo "  SETUP COMPLETE!"
echo "========================================"
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python train.py --model MODEL_NAME --dataset data.json"
echo ""
echo "Quick start:"
echo "  python batch_convert.py                    # Convert text files in data/"
echo "  python train.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d dataset.json"
echo ""

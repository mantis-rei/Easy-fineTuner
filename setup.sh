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

# Update system
echo "Updating system..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget curl

# Create virtual environment
echo "Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
echo "Installing training libraries..."
pip install -q transformers datasets peft trl bitsandbytes accelerate sentencepiece protobuf

# Verify GPU
echo ""
echo "Checking GPU..."
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')"
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"

echo ""
echo "========================================"
echo "  SETUP COMPLETE!"
echo "========================================"
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python train.py --model MODEL_NAME --dataset data.json"
echo ""
echo "Example:"
echo "  python train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset training_dataset.json"
echo ""

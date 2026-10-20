#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "  3D Scene Understanding & Architecture Studio - Environment Setup"
echo "======================================================================"

# 1. Create Virtual Environment if not exists
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python 3.10 virtual environment in .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch with CUDA support
echo "[2/4] Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install Requirements
echo "[3/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Clone and install VGGT submodule if not present
echo "[4/4] Setting up VGGT foundation model..."
if [ ! -d "vggt" ]; then
    git clone https://github.com/facebookresearch/vggt.git vggt
fi
pip install -e vggt

echo ""
echo "======================================================================"
echo "  ✓ Setup successfully completed!"
echo "  Activate your environment with: source .venv/bin/activate"
echo "  Launch App-Architect with:      python app_architect.py"
echo "  Launch 3D Navigator with:       python app.py"
echo "======================================================================"

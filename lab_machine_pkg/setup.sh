#!/bin/bash
# Setup script for the energy-analysis pipeline on Ubuntu 20.04 lab machine.
# Creates a Python 3.10 venv and installs pinned dependencies.
#
# Prerequisites (must be done before running this script):
#   - Python 3.10 must be installed. On Ubuntu 20.04 (which ships Python 3.8),
#     run these once:
#       sudo apt install -y software-properties-common
#       sudo add-apt-repository -y ppa:deadsnakes/ppa
#       sudo apt update
#       sudo apt install -y python3.10 python3.10-venv python3.10-dev
#   - You must have an NVIDIA GPU with a working driver. Confirm with `nvidia-smi`.

set -e  # exit on any error

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.10 venv at $VENV_DIR..."
    python3.10 -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR. Reusing."
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing pinned requirements (this may take 5-10 minutes for torch and friends)..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Next steps:"
echo "  1. Verify GPU: source $VENV_DIR/bin/activate && python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\")'"
echo "  2. Start your energy logger (whatever Tim's tool is)."
echo "  3. Run: ./run_all.sh"

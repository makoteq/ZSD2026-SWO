#!/bin/bash

# Hardware Deployment Setup Script for Raspberry Pi 4 B
# This script automates OS updates, CPU performance tuning,
# Miniforge3 (Conda) installation, and environment configuration.
set -e

ENV_NAME="zsd"
DEB_PACKAGE="libedgetpu1-max_16.0tf2.19.1-1.bookworm_arm64.deb"
DEB_URL="https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-max_16.0tf2.19.1-1.bookworm_arm64.deb"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"

echo "========================================================"
echo "=== 1. Checking and Installing Conda (Miniforge3) ==="
echo "========================================================"

if ! command -v conda &> /dev/null; then
    echo "Conda was not found. Downloading Miniforge3 for ARM64..."
    wget -q "$MINIFORGE_URL" -O Miniforge3.sh

    echo "Installing Miniforge3 to $HOME/miniforge3..."
    bash Miniforge3.sh -b -p "$HOME/miniforge3"
    rm -f Miniforge3.sh

    echo "Initializing Conda for Bash shell..."
    "$HOME/miniforge3/bin/conda" init bash

    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "Conda is already installed."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "========================================================"
echo "=== 2. Updating OS Packages and Dependencies ==="
echo "========================================================"
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl wget python3-pip python3-pil python3-numpy

echo "========================================================"
echo "=== 3. Installing Edge TPU High-Performance Runtime ==="
echo "========================================================"

if dpkg -l | grep -q libedgetpu1-std; then
    echo "Removing legacy libedgetpu1-std package..."
    sudo apt-get remove --purge libedgetpu1-std -y
fi

if [ ! -f "$DEB_PACKAGE" ]; then
    echo "Downloading customized libedgetpu1-max package..."
    wget -q "$DEB_URL"
fi

echo "Installing libedgetpu1-max package..."
sudo dpkg -i "$DEB_PACKAGE"
rm -f "$DEB_PACKAGE"

echo "========================================================"
echo "=== 4. Configuring CPU Performance Governor ==="
echo "========================================================"

if ! grep -q "scaling_governor" /etc/rc.local; then
    echo "Adding scaling_governor persistent configuration to /etc/rc.local..."
    sudo sed -i -e '$i \echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor\n' /etc/rc.local
fi

echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo "Active CPU Governor:" $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)

echo "========================================================"
echo "=== 5. Setting up Python Virtual Environment ==="
echo "========================================================"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation..."
else
    echo "Creating virtual environment from environment.yaml..."
    conda env create -f environment.yaml
fi

echo "Activating virtual environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Upgrading ultralytics library..."
pip install -U "ultralytics[export]"

echo "========================================================"
echo "=== Setup completed successfully! ==="
echo "========================================================"
echo "Post-install instructions:"
echo "1. UNPLUG AND REPLUG your Google Coral device into a BLUE USB 3.0 port to apply the 500 MHz driver."
echo "2. Close this terminal and open a new one, or run: source ~/.bashrc"
echo "3. To activate the environment manually: conda activate $ENV_NAME"
echo "========================================================"

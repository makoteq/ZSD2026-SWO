#!/bin/bash

set -e

ENV_NAME="zsd"

echo "=== Checking Conda ==="

if ! command -v conda &> /dev/null
then
    echo "Conda is not installed."
    exit 1
fi

echo "=== Initializing Conda ==="
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating environment from environment.yaml ==="
conda env create -f environment.yaml

echo "=== Activating environment ==="
conda activate $ENV_NAME

echo "=== Environment created successfully ==="

echo "To activate later use:"
echo "conda activate $ENV_NAME"
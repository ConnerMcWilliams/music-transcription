#!/usr/bin/env bash
set -e

MINI_DIR="$(pwd)/../conda"
INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

mkdir -p "$MINI_DIR"

if [ -f "$INSTALLER" ]; then
    echo "$INSTALLER already exists"
else
    echo "Downloading Miniconda installer..."
    wget "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$INSTALLER"
fi

if [ -d "$MINI_DIR" ] && [ -f "$MINI_DIR/bin/conda" ]; then
    echo "Miniconda already installed at $MINI_DIR"
else
    echo "Installing Miniconda..."
    bash "$INSTALLER" -b -p "$MINI_DIR"
fi

source "$MINI_DIR/etc/profile.d/conda.sh"

conda env create -f environment.yml || conda env update -f environment.yml
conda activate mamba-amt

echo "Environment activated: $CONDA_DEFAULT_ENV"
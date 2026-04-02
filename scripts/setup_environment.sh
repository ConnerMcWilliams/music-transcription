#!/usr/bin/env bash
set -e

CURRENT_DIR="$(pwd)"
MINI_DIR="$CURRENT_DIR/../conda"
INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
EVIRONMENT_DIR="$CURRENT_DIR/../environment.yml"

mkdir -p "$MINI_DIR"

if [ -f "$INSTALLER" ]; then
    echo "$INSTALLER already exists"
else
    echo "Downloading Miniconda installer..."
    wget "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$INSTALLER"
fi

if [ -f "$MINI_DIR/bin/conda" ]; then
    echo "Miniconda already installed at $MINI_DIR"
else
    echo "Installing Miniconda..."
    bash "$INSTALLER" -u -b -p "$MINI_DIR"
fi

source "$MINI_DIR/etc/profile.d/conda.sh"

conda env create -f $EVIRONMENT_DIR || conda env update -f $EVIRONMENT_DIR
conda activate mamba-amt

echo "Environment activated: $CONDA_DEFAULT_ENV"
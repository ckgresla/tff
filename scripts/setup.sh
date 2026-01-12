#!/bin/bash

set -e
eval "$(conda shell.bash hook)"

# rebuild env from scratch, nuke old env if exists
conda env remove -p /opt/miniconda/envs/tff -y 2>/dev/null || true

conda create -p /opt/miniconda/envs/tff python=3.12 -y
conda activate tff
echo -e "using python: $(which python)\n\n"

# Install main package + its deps
pip install -e ".[dev]"


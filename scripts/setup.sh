#!/bin/bash
# a dependable conda based setup process
set -e
eval "$(conda shell.bash hook)"

conda create -n magic python=3.12 -y
conda activate magic
echo -e "using python: $(which python)\n\n"

pip install -e ".[dev]"

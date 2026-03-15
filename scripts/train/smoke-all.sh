#!/bin/bash
# Smoke test all three model types (DP rank 4, ~20 steps each)
set -e

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,4,5

echo "=== Smoke test: RegularTransformer ==="
python -m choosy.train \
    model=regular-toy \
    training=smoke \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/smoke-regular

echo ""
echo "=== Smoke test: LoopedTransformer ==="
python -m choosy.train \
    model=looped-toy \
    training=smoke \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/smoke-looped

echo ""
echo "=== Smoke test: ChoosyTransformer ==="
python -m choosy.train \
    model=choosy-toy \
    training=smoke \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/smoke-choosy

echo ""
echo "=== All smoke tests passed ==="

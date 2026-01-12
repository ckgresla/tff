#!/bin/bash
# Toy training script for testing a small transformer
# Single GPU, no data parallelism

set -e
unset LD_LIBRARY_PATH

# Force single GPU (GPU 0)
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda

# Optional: Enable JAX debug mode for development
# export JAX_DEBUG_NANS=True
# export JAX_DISABLE_JIT=False

echo "Toy Training Script"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "JAX Platform: $JAX_PLATFORMS"
echo ""

# Run training with small config
python -m tff.train_toy

echo ""
echo "Training complete!"

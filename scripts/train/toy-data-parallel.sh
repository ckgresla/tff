#!/bin/bash
# Toy training script with DATA PARALLELISM across multiple GPUs
# This demonstrates clean data parallel training using JAX sharding

set -e
unset LD_LIBRARY_PATH

# Configure GPUs to use for data parallelism via CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,4,5
export JAX_PLATFORMS=cuda

# some optimizations
# use the compilation cache, faster startups
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
# memory preallocation (recommended?)
# export XLA_PYTHON_CLIENT_PREALLOCATE=true

# Print configuration
echo "============================================"
echo "Toy Training - DATA PARALLEL"
echo "============================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "JAX Platform: $JAX_PLATFORMS"
echo "============================================"
echo ""

# Run training with toy config + data parallel
export WANDB_PROJECT="tff-development"
export WANDB_NAME="wilburwright-v8"
python -m tff.train model=toy training=toy training.data_parallel=true optimizer=sgd optimizer.nesterov=True optimizer.momentum=0.4

echo ""
echo "============================================"
echo "Data parallel training complete!"
echo "============================================"

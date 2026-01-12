#!/bin/bash
# Toy training script with DATA PARALLELISM across multiple GPUs
# This demonstrates clean data parallel training using JAX sharding

set -e
unset LD_LIBRARY_PATH

# Configure GPUs to use for data parallelism via CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,4,5
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

# Run training with data_parallel enabled.
python -m tff.train_toy --data_parallel=True

echo ""
echo "============================================"
echo "Data parallel training complete!"
echo "============================================"

# Transformer Former (TFF)

**Could a transformer learn to form itself?**

## Concept

Traditional transformers apply layers in a fixed, sequential order. This project explores **dynamic layer routing** - instead of a fixed sequence, the model:

1. Maintains a **pool of N transformer layers**
2. At each computational step, **routes** the current representation to one of the layers
3. Applies the selected layer and repeats for K steps

This allows the model to learn adaptive computation paths through the layer pool, similar to Mixture of Experts but for entire transformer layers.

## Implementation

Built with JAX and Equinox for clean, functional code:

- **`tff/components.py`**: Core transformer building blocks (attention, FFN, transformer layer)
- **`tff/routing.py`**: Dynamic routing mechanisms (LayerRouter, LayerPool)
- **`tff/model.py`**: Main DynamicTransformer model

### Key Features

- **Layer Pool**: A set of transformer layers that can be applied in any order
- **Router**: MLP-based router that selects which layer to apply based on current representation
- **Multi-step Routing**: Apply K routing steps to build adaptive computation paths
- **Balanced Routing Loss**: Auxiliary loss to encourage uniform layer usage
- **Training vs Inference**: Stochastic routing (sampling) during training, deterministic (argmax) during inference

## Usage

```python
import jax.random as jr
from tff import DynamicTransformer

# Create model
key = jr.PRNGKey(42)
model = DynamicTransformer(
    vocab_size=1000,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_pool_layers=6,  # Pool of 6 layers
    num_steps=4,        # 4 routing steps
    key=key,
)

# Forward pass
tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
logits, info = model(tokens, key=key, training=True)

# Routing information
layer_choices = info["layer_choices"]  # [batch, num_steps]
router_logits = info["router_logits"]  # [batch, num_steps, num_layers]

# Compute auxiliary routing loss
routing_loss = model.compute_routing_loss(router_logits)
```

## Testing

```bash
# Quick test of GPU setup
bash scripts/test-gpu.sh

# Run all component tests
bash scripts/run-test.sh

# Or run tests manually
python -m tff.test_init
python -m tff.test_simple
```

## Training

### Single GPU Training

```bash
# Toy training (small model, quick test)
./scripts/train/toy.sh

# Or run directly
python -m tff.train_toy
```

### Multi-GPU Data Parallel Training

The codebase implements clean, modern **data parallelism** using JAX sharding and Equinox.

```bash
# Train on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m tff.train_toy --data_parallel

# Train on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m tff.train_toy --data_parallel

# Or use the provided script (edit GPU selection inside)
./scripts/train/toy_data_parallel.sh
```

**Key Features:**
- ✨ SIMD-like code - no scattered conditionals
- ✨ Computation follows data - JAX handles distribution
- ✨ Same code works for 1 GPU or N GPUs
- ✨ Detailed diagnostic output for verification

**Requirements:**
- Batch size must be divisible by number of GPUs
- Example: `batch_size=8` works with 1, 2, 4, or 8 GPUs

**See [DATA_PARALLEL.md](DATA_PARALLEL.md) for detailed documentation.**

### Verify Data Parallel Setup

```bash
# Test on single GPU
CUDA_VISIBLE_DEVICES=0 python -m tff.examples.verify_data_parallel

# Test on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m tff.examples.verify_data_parallel --batch_size 8
```

## Environment Setup

```bash
# Run setup script (recommended)
bash scripts/setup.sh

# This will:
# - Create conda environment at /opt/miniconda/envs/tff
# - Install all dependencies
# - Configure CUDA library paths automatically

# Activate environment
conda activate tff

# The activation script will automatically set LD_LIBRARY_PATH for JAX

# Verify JAX detects GPUs
python -c "import jax; print(jax.devices())"

# Test matmul works
python -c "import jax.numpy as jnp; x = jnp.ones((10,10)); print(jnp.matmul(x, x).shape)"
```

## Next Steps

- [ ] Train on a language modeling task (e.g., character-level text)
- [ ] Analyze learned routing patterns
- [ ] Experiment with different pool sizes and routing strategies
- [ ] Visualize which layers are used in different contexts
- [ ] Compare to baseline sequential transformer
- [ ] Try conditional routing (based on input tokens, not just representation)
- [ ] Explore hierarchical routing (routing to groups of layers)

## Research Questions

1. **Do models learn specialized layers?** Will certain layers become specialists for different types of operations?
2. **How does routing change during training?** Does the model converge to fixed paths or maintain stochasticity?
3. **Can this improve sample efficiency?** Does adaptive routing allow better learning with less data?
4. **What about computation efficiency?** Can we skip layers dynamically for simpler inputs?

## Architecture Diagram

```
Input Tokens
     ↓
  Embedding + Positional
     ↓
  ┌─────────────────┐
  │  Routing Step 1  │
  │  ┌──────────┐   │
  │  │  Router  │→ Select Layer from Pool
  │  └──────────┘   │
  │       ↓         │
  │  Apply Layer    │
  └─────────────────┘
     ↓
  ┌─────────────────┐
  │  Routing Step 2  │
  │  ┌──────────┐   │
  │  │  Router  │→ Select Layer from Pool
  │  └──────────┘   │
  │       ↓         │
  │  Apply Layer    │
  └─────────────────┘
     ↓
    ...
     ↓
  Output Projection
     ↓
  Logits
```

## Notes

- JAX is pinned to version 0.6.0 for CUDA compatibility
- **GPU Fix Applied**: The setup script automatically configures LD_LIBRARY_PATH for CUDA libraries
- If you still have issues, see `GPU_ISSUE.md` for manual configuration
- **Fallback**: Use `JAX_PLATFORMS=cpu` to run on CPU if needed
- Router uses mean-pooled representation to make routing decisions
- Each batch element can take a different path through the layer pool

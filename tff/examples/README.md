# TFF Examples

Example scripts demonstrating various features of the TFF library.

## Training Metrics Analysis

### `analyze_metrics.py`

Analyze training runs and compare results.

**Basic analysis:**
```bash
python tff/examples/analyze_metrics.py checkpoints
```

**Compare two runs:**
```bash
python tff/examples/analyze_metrics.py compare checkpoints/run1 checkpoints/run2
```

**Export to CSV:**
```bash
python tff/examples/analyze_metrics.py export checkpoints metrics.csv
```

**Output example:**
```
================================================================================
Training Summary
================================================================================

Model:
  Parameters:    524,288
  d_model:       128
  Layers:        2
  Heads:         4

Training:
  Total steps:   1,000
  Total tokens:  1,024,000 (1.02M)
  Batch size:    8
  Seq length:    128
  Time:          2.34 minutes

Best Validation:
  Loss:          2.1234
  BPC:           3.0645
  Step:          850

Final Validation:
  Loss:          2.1456
  BPC:           3.0965

Throughput:
  Tokens/sec:    7,291
  Steps/sec:     56.82
================================================================================
```

## Checkpoint Loading

### `load_checkpoint.py`

Examples of loading and using saved model checkpoints.

**Run all examples:**
```bash
python tff/examples/load_checkpoint.py
```

**Key functions demonstrated:**
- Loading best/final/specific checkpoints
- Using models for inference
- Accessing checkpoint configurations
- Conceptual example of continuing training

## Python API Usage

### Loading Metrics

```python
from tff.metrics import load_metrics, load_training_info, print_training_summary

# Quick summary
print_training_summary("checkpoints")

# Detailed analysis
metrics = load_metrics("checkpoints")
info = load_training_info("checkpoints")

# Access specific metrics
for m in metrics:
    if m.val_loss is not None:
        print(f"Step {m.step}: Val BPC = {m.val_bpc:.4f}")

# Get throughput stats
print(f"Final throughput: {metrics[-1].tokens_per_second:,.0f} tokens/sec")
print(f"Total tokens: {info.total_tokens:,}")
```

### Loading Checkpoints

```python
from tff.checkpoint import load_best_checkpoint, load_checkpoint

# Load best model
model, config = load_best_checkpoint("checkpoints")

# Load specific step
model, config = load_checkpoint("checkpoints", "checkpoint-001000")

# Use for inference
import jax.random as jr
import jax.numpy as jnp

key = jr.PRNGKey(0)
inputs = jr.randint(key, shape=(1, 256), minval=0, maxval=256)
logits = model(inputs, key=None)  # No dropout for inference
predictions = jnp.argmax(logits, axis=-1)
```

## Metrics File Formats

### `training-metrics.jsonl`

JSONL format (one JSON object per line) for easy streaming:

```json
{"step": 0, "loss": 5.1234, "bpc": 7.3908, "learning_rate": 0.0003, "tokens_in_batch": 1024, "total_tokens_seen": 1024, "elapsed_seconds": 0.52, "steps_per_second": 1.92, "tokens_per_second": 1966.15}
{"step": 20, "loss": 4.8765, "bpc": 7.0315, "learning_rate": 0.0003, "tokens_in_batch": 1024, "total_tokens_seen": 21504, "elapsed_seconds": 10.23, "steps_per_second": 2.05, "tokens_per_second": 2102.44}
{"step": 100, "loss": 3.2145, "bpc": 4.6375, "learning_rate": 0.0003, "tokens_in_batch": 1024, "total_tokens_seen": 103424, "elapsed_seconds": 52.14, "steps_per_second": 1.92, "tokens_per_second": 1984.23, "val_loss": 3.2567, "val_bpc": 4.6984}
```

### `training-info.json`

Summary JSON with final statistics:

```json
{
  "model_params": 524288,
  "d_model": 128,
  "num_layers": 2,
  "num_heads": 4,
  "batch_size": 8,
  "seq_len": 128,
  "total_steps": 1000,
  "total_tokens": 1024000,
  "total_time_seconds": 140.56,
  "best_val_loss": 2.1234,
  "best_val_bpc": 3.0645,
  "best_val_step": 850,
  "final_val_loss": 2.1456,
  "final_val_bpc": 3.0965,
  "metrics_file": "training-metrics.jsonl"
}
```

## Creating Custom Analyses

You can easily create custom analyses by loading the metrics:

```python
import matplotlib.pyplot as plt
from tff.metrics import load_metrics

# Load metrics
metrics = load_metrics("checkpoints")

# Extract data
steps = [m.step for m in metrics]
losses = [m.loss for m in metrics]
bpcs = [m.bpc for m in metrics]

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(steps, bpcs, label='Training BPC')

# Add validation points
val_steps = [m.step for m in metrics if m.val_bpc is not None]
val_bpcs = [m.val_bpc for m in metrics if m.val_bpc is not None]
plt.scatter(val_steps, val_bpcs, color='red', label='Validation BPC', zorder=5)

plt.xlabel('Training Step')
plt.ylabel('Bits Per Character (BPC)')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
```

## Tips

1. **JSONL format** - Each line is a valid JSON object, easy to stream and parse
2. **Token counting** - Automatic tracking of total tokens processed
3. **Throughput metrics** - tokens/sec and steps/sec computed automatically
4. **Validation tracking** - Best validation automatically recorded
5. **Export to CSV** - Easy integration with pandas, R, Excel, etc.

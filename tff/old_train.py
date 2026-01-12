"""Training script for byte-level language modeling on enwik8."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
from functools import partial
from typing import Optional

from tff.model import DynamicTransformer
from tff.data import Enwik8Dataset


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss."""
    # logits: [batch, seq, vocab]
    # targets: [batch, seq]
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)

    # Gather log probs for targets
    target_log_probs = jnp.take_along_axis(
        log_probs,
        targets_flat[:, None],
        axis=1
    ).squeeze(1)

    # Return negative log likelihood
    return -jnp.mean(target_log_probs)


@eqx.filter_jit
def compute_loss(model, inputs, targets, key, training=True):
    """Compute loss for a batch."""
    # Forward pass
    logits, info = model(inputs, key=key, training=training)

    # Main loss
    main_loss = cross_entropy_loss(logits, targets)

    # Routing balance loss
    routing_loss = model.compute_routing_loss(info["router_logits"])

    # Total loss
    total_loss = main_loss + routing_loss

    return total_loss, {
        "loss": main_loss,
        "routing_loss": routing_loss,
        "total_loss": total_loss,
    }


@eqx.filter_jit
def train_step(model, opt_state, inputs, targets, key):
    """Single training step."""
    # Compute loss and gradients
    (loss, metrics), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, inputs, targets, key, training=True)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, metrics


@eqx.filter_jit
def eval_step(model, inputs, targets, key):
    """Single evaluation step."""
    _, metrics = compute_loss(model, inputs, targets, key, training=False)
    return metrics


def train(
    model_config: dict,
    data_path: str = "/raid/datasets/enwiki8.zip",
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    num_steps: int = 10000,
    eval_every: int = 500,
    seed: int = 42,
):
    """
    Train the dynamic transformer on enwik8.

    Args:
        model_config: Model configuration dict
        data_path: Path to enwik8.zip
        batch_size: Batch size
        learning_rate: Learning rate
        num_steps: Number of training steps
        eval_every: Evaluate every N steps
        seed: Random seed
    """
    print("=" * 70)
    print("Training Dynamic Routing Transformer on enwik8")
    print("=" * 70)

    # JAX device info
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    train_data = Enwik8Dataset(data_path, seq_len=seq_len, split="train")
    valid_data = Enwik8Dataset(data_path, seq_len=seq_len, split="valid")

    # Initialize model
    print("\nInitializing model...")
    key = jr.PRNGKey(seed)
    model_key, train_key = jr.split(key)

    model = DynamicTransformer(**model_config, key=model_key)
    print(f"Model initialized with {model_config['num_pool_layers']} layers in pool")
    print(f"Applying {model_config['num_steps']} routing steps per forward pass")

    # Initialize optimizer
    print(f"\nSetting up optimizer (lr={learning_rate})...")
    global optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Training loop
    print("\nStarting training...")
    print(f"Batch size: {batch_size}")
    print(f"Total steps: {num_steps}")
    print("-" * 70)

    rng = np.random.default_rng(seed)
    train_batches = train_data.iterate_batches(batch_size, rng)

    for step in range(num_steps):
        # Get batch
        inputs, targets = next(train_batches)

        # Train step
        train_key, step_key = jr.split(train_key)
        model, opt_state, metrics = train_step(
            model, opt_state, inputs, targets, step_key
        )

        # Log
        if step % 100 == 0:
            print(
                f"Step {step:5d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Routing: {metrics['routing_loss']:.6f}"
            )

        # Evaluate
        if step % eval_every == 0 and step > 0:
            print(f"\n{'Evaluating':-^70}")

            # Eval on a few batches
            eval_rng = np.random.default_rng(seed + 1)
            eval_batches = valid_data.iterate_batches(batch_size, eval_rng, num_batches=20)

            eval_losses = []
            for eval_inputs, eval_targets in eval_batches:
                train_key, eval_key = jr.split(train_key)
                eval_metrics = eval_step(model, eval_inputs, eval_targets, eval_key)
                eval_losses.append(float(eval_metrics['loss']))

            avg_eval_loss = np.mean(eval_losses)
            print(f"Validation Loss: {avg_eval_loss:.4f}")
            print(f"Validation BPC: {avg_eval_loss / np.log(2):.4f}")
            print("-" * 70 + "\n")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import os
    # Force GPU usage
    os.environ['JAX_PLATFORMS'] = 'gpu'

    # Model configuration - smaller for quick testing
    config = {
        "vocab_size": 256,  # Byte-level (0-255)
        "d_model": 256,  # Smaller
        "num_heads": 8,
        "d_ff": 1024,  # Smaller
        "num_pool_layers": 6,  # Pool of 6 layers
        "num_steps": 4,  # 4 routing steps
        "max_seq_len": 512,
        "dropout_rate": 0.1,
        "router_hidden_size": 128,
        "router_temperature": 1.0,
    }

    # Sequence length from data loader
    seq_len = 256

    # Train
    model = train(
        model_config=config,
        batch_size=8,  # Smaller batch
        learning_rate=3e-4,
        num_steps=5000,
        eval_every=250,
    )

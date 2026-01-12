"""Example: How to load a checkpoint with Equinox.

This script demonstrates:
1. Loading a saved model checkpoint
2. Using the loaded model for inference
3. Continuing training from a checkpoint
"""

import jax.numpy as jnp
import jax.random as jr

from tff.checkpoint import load_checkpoint, load_best_checkpoint, list_checkpoints
from tff.train import train


def example_load_and_use():
    """Load a checkpoint and use it for inference."""
    print("=" * 80)
    print("Example 1: Loading a checkpoint")
    print("=" * 80)

    # List available checkpoints
    checkpoint_dir = "checkpoints"
    available = list_checkpoints(checkpoint_dir)
    print(f"\nAvailable checkpoints in '{checkpoint_dir}':")
    for name in available:
        print(f"  - {name}")

    # Load the best model
    print("\nLoading best model checkpoint...")
    model, config = load_best_checkpoint(checkpoint_dir)

    print(f"\nLoaded model with {model.count_parameters():,} parameters")
    print(config.summary())

    # Use model for inference
    print("\n" + "=" * 80)
    print("Running inference...")
    print("=" * 80)

    # Create a sample input (batch of 1, sequence length from config)
    key = jr.PRNGKey(0)
    sample_input = jr.randint(
        key,
        shape=(1, config.data.seq_len),
        minval=0,
        maxval=config.model.vocab_size
    )

    # Forward pass (no dropout during inference)
    logits = model(sample_input, key=None)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Get predictions (greedy decoding)
    predictions = jnp.argmax(logits, axis=-1)
    print(f"Predictions shape: {predictions.shape}")


def example_load_specific_checkpoint():
    """Load a checkpoint from a specific training step."""
    print("\n" + "=" * 80)
    print("Example 2: Loading a specific checkpoint")
    print("=" * 80)

    checkpoint_dir = "checkpoints"
    checkpoint_name = "checkpoint_001000"  # Step 1000

    print(f"\nLoading checkpoint: {checkpoint_name}")
    model, config = load_checkpoint(checkpoint_dir, checkpoint_name)

    print(f"Loaded model from step 1000")
    print(f"Parameters: {model.count_parameters():,}")


def example_continue_training():
    """Continue training from a checkpoint."""
    print("\n" + "=" * 80)
    print("Example 3: Continue training from checkpoint")
    print("=" * 80)

    # Load checkpoint
    checkpoint_dir = "checkpoints"
    model, config = load_best_checkpoint(checkpoint_dir)
    print(f"\nLoaded checkpoint, resuming training...")

    # Modify config for continued training
    # Note: Pydantic configs are frozen, so we need to create a new one
    from tff.config import ExperimentConfig, TrainingConfig

    new_config = ExperimentConfig(
        model=config.model,  # Keep same model architecture
        data=config.data,    # Keep same data settings
        training=TrainingConfig(
            learning_rate=config.training.learning_rate * 0.1,  # Lower LR for fine-tuning
            num_steps=1000,  # Train for 1000 more steps
            eval_every=100,
            log_every=50,
            seed=config.training.seed + 1,  # Different seed
            checkpoint_dir="checkpoints_continued",
            save_every=500,
        ),
    )

    print("\nNew training config:")
    print(new_config.summary())

    # Continue training
    # Note: You would need to modify the train() function to accept
    # an optional pre-initialized model for this to work.
    # For now, this is just a conceptual example.
    print("\n(In practice, you'd modify train() to accept a pre-loaded model)")


def main():
    """Run all examples."""
    # Example 1: Basic loading
    example_load_and_use()

    # Example 2: Load specific checkpoint
    example_load_specific_checkpoint()

    # Example 3: Continue training (conceptual)
    example_continue_training()

    print("\n" + "=" * 80)
    print("Summary: Loading checkpoints with Equinox")
    print("=" * 80)
    print("""
Key points:
1. Use load_checkpoint(dir, name) to load any checkpoint
2. Use load_best_checkpoint(dir) for the best model
3. Checkpoints save both model weights (.eqx) and config (.json)
4. Model architecture must match the saved checkpoint
5. The config is automatically loaded and used to reconstruct the model

Quick reference:
    from tff.checkpoint import load_checkpoint, load_best_checkpoint

    # Load best model
    model, config = load_best_checkpoint("checkpoints")

    # Load specific step
    model, config = load_checkpoint("checkpoints", "checkpoint_001000")

    # Use for inference (no dropout)
    logits = model(input_tokens, key=None)
    """)


if __name__ == "__main__":
    main()

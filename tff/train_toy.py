"""Toy training script for quick testing with a small model."""

import os
import fire
from pathlib import Path
from tff import train, ExperimentConfig, ModelConfig, DataConfig, TrainingConfig


def main(data_parallel: bool = False) -> None:
    """Run toy training with very small model for quick testing.

    Args:
        data_parallel: Enable data parallelism across all visible GPUs.
                      Use CUDA_VISIBLE_DEVICES to control which GPUs are used.
                      Example: CUDA_VISIBLE_DEVICES=0,1 python -m tff.train_toy --data_parallel
    """
    # Force GPU usage
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')

    # Create a tiny config for rapid iteration
    # Note: For data parallelism, batch_size must be divisible by num_devices
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,      # Byte-level
            d_model=128,         # Very small
            num_layers=4,        # Just a few layers
            num_heads=4,         # and heads
            d_ff=512,            # 4x d_model
            max_seq_len=256,
            dropout_rate=0.1,
        ),
        data=DataConfig(
            data_path="/raid/datasets/enwiki8.zip",
            seq_len=256,       # Short sequences
            batch_size=64,     # something "divisible"
        ),
        training=TrainingConfig(
            learning_rate=2e-4,
            num_steps=10000,      # something like a quick test
            eval_every=1000,      # Eval often
            log_every=100,        # Log frequently
            seed=42,
            checkpoint_dir="checkpoints/toy",
            save_every=1000,
            eval_on_start=True,
            data_parallel=data_parallel,  # Pass to config
            # NOTE: all wandb config is done with env vars!
        ),
    )

    print("\nTOY TRAINING - Small model for quick testing")
    print(f"Data parallelism: {'ENABLED' if data_parallel else 'DISABLED'}")
    print("This will train a tiny 8-layer transformer for 25k steps.")
    if data_parallel:
        print("Batch will be split across all visible GPUs.")
    if config.training.wandb_project:
        print(f"W&B logging: ENABLED (project: {config.training.wandb_project})")

    # Train
    _ = train(config)

    checkpoint_dir = Path(config.training.checkpoint_dir).resolve()
    print(f"\nToy training complete!")
    print(f"Checkpoints saved to: '{checkpoint_dir}'")
    print("\nLoad the model with:")
    print("  from tff.checkpoint import load_best_checkpoint")
    print(f"  model, cfg = load_best_checkpoint('{checkpoint_dir}')\n")


if __name__ == "__main__":
    fire.Fire(main)

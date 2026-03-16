"""Checkpoint saving and loading utilities for JAX/Equinox models."""

import logging
from pathlib import Path
from typing import Optional

import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray

from choosy.config import ExperimentConfig

log = logging.getLogger(__name__)


def save_checkpoint(
    model,
    config: ExperimentConfig,
    checkpoint_dir: Path,
    name: str = "checkpoint",
) -> None:
    """Save model checkpoint with configuration.

    Creates two files:
    - {name}.eqx: Equinox model weights
    - {name}_config.json: Experiment configuration

    Args:
        model: Model to save
        config: Experiment configuration
        checkpoint_dir: Directory to save checkpoint
        name: Name prefix for checkpoint files (e.g., "best", "checkpoint_001000")
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = checkpoint_dir / f"{name}.eqx"
    eqx.tree_serialise_leaves(model_path, model)

    # Save configuration
    config_path = checkpoint_dir / f"{name}-config.json"
    config.save_json(config_path)

    log.info("Saved checkpoint to '%s/%s.*'", checkpoint_dir, name)


def load_checkpoint(
    checkpoint_dir: Path,
    name: str = "checkpoint",
    key: Optional[PRNGKeyArray] = None,
) -> tuple[eqx.Module, ExperimentConfig]:
    """Load model checkpoint with configuration.

    Args:
        checkpoint_dir: Directory containing checkpoint
        name: Name prefix for checkpoint files
        key: Optional PRNG key for model initialization (will use seed from config if None)

    Returns:
        Tuple of (model, config)

    Example:
        >>> model, config = load_checkpoint("checkpoints", "best")
        >>> # Now you can use the model for inference or continued training
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load configuration
    config_path = checkpoint_dir / f"{name}-config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = ExperimentConfig.load_json(config_path)

    # Initialize model with same architecture
    from choosy.train import create_model
    if key is None:
        key = jr.PRNGKey(config.training.seed)

    model = create_model(config.model, key=key)

    # Load model weights
    model_path = checkpoint_dir / f"{name}.eqx"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = eqx.tree_deserialise_leaves(model_path, model)

    log.info("Loaded checkpoint from '%s/%s.*'", checkpoint_dir, name)
    return model, config


def list_checkpoints(checkpoint_dir: Path) -> list[str]:
    """List all available checkpoints in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        List of checkpoint names (without file extensions)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    # Find all .eqx files and extract their names
    checkpoint_names = set()
    for path in checkpoint_dir.glob("*.eqx"):
        name = path.stem
        # Check if corresponding config exists
        config_path = checkpoint_dir / f"{name}-config.json"
        if config_path.exists():
            checkpoint_names.add(name)

    return sorted(checkpoint_names)


# Example usage functions
def load_best_checkpoint(checkpoint_dir: Path) -> tuple[eqx.Module, ExperimentConfig]:
    """Load the best model checkpoint."""
    return load_checkpoint(checkpoint_dir, "best-model")


def load_final_checkpoint(checkpoint_dir: Path) -> tuple[eqx.Module, ExperimentConfig]:
    """Load the final model checkpoint."""
    return load_checkpoint(checkpoint_dir, "final-model")


def load_step_checkpoint(
    checkpoint_dir: Path,
    step: int,
) -> tuple[eqx.Module, ExperimentConfig]:
    """Load checkpoint from a specific training step."""
    name = f"checkpoint-{step:06d}"
    return load_checkpoint(checkpoint_dir, name)

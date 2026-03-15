"""Training metrics tracking and logging."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import jax.numpy as jnp


@dataclass
class StepMetrics:
    """Metrics for a single training step."""

    step: int
    loss: float
    bpc: float
    learning_rate: float

    # Token counting
    tokens_in_batch: int
    total_tokens_seen: int

    # Timing
    elapsed_seconds: float
    steps_per_second: float
    tokens_per_second: float

    # Optional validation metrics
    val_loss: Optional[float] = None
    val_bpc: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TrainingInfo:
    """Complete training information and metrics history."""

    # Configuration summary
    model_params: int
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    seq_len: int

    # Training progress
    total_steps: int
    total_tokens: int
    total_time_seconds: float

    # Best metrics
    best_val_loss: float
    best_val_bpc: float
    best_val_step: int

    # Final metrics
    final_val_loss: float
    final_val_bpc: float

    # Per-step metrics (stored separately for efficiency)
    metrics_file: str = "training-metrics.jsonl"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        """Save training info to JSON.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "TrainingInfo":
        """Load training info from JSON.

        Args:
            path: Path to JSON file

        Returns:
            TrainingInfo instance
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MetricsTracker:
    """Track and save training metrics."""

    def __init__(
        self,
        output_dir: Path,
        batch_size: int,
        seq_len: int,
        model_params: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
    ):
        """Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics
            batch_size: Training batch size
            seq_len: Sequence length
            model_params: Number of model parameters
            d_model: Model dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens_per_batch = batch_size * seq_len

        self.model_params = model_params
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Tracking state
        self.total_tokens = 0
        self.best_val_loss = float('inf')
        self.best_val_bpc = float('inf')
        self.best_val_step = 0

        # Open metrics file for appending (JSONL format)
        self.metrics_path = self.output_dir / "training-metrics.jsonl"
        self.metrics_file = open(self.metrics_path, 'w')

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        elapsed_seconds: float,
        val_loss: Optional[float] = None,
    ) -> StepMetrics:
        """Log metrics for a training step.

        IMPORTANT: All arguments must be Python native types (int, float), not JAX arrays.
        Convert JAX arrays before calling this method.

        Args:
            step: Current step number (Python int)
            loss: Training loss (Python float)
            learning_rate: Current learning rate (Python float)
            elapsed_seconds: Time elapsed since training start (Python float)
            val_loss: Optional validation loss (Python float or None)

        Returns:
            StepMetrics object with computed metrics
        """
        # Update token count
        self.total_tokens += self.tokens_per_batch

        # Compute derived metrics
        bpc = loss / jnp.log(2)
        bpc = float(bpc)
        steps_per_second = (step + 1) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        tokens_per_second = self.total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

        # Create metrics object
        val_bpc = None
        if val_loss is not None:
            val_bpc = val_loss / jnp.log(2)
            val_bpc = float(val_bpc)

        metrics = StepMetrics(
            step=step,
            loss=loss,
            bpc=bpc,
            learning_rate=learning_rate,
            tokens_in_batch=self.tokens_per_batch,
            total_tokens_seen=self.total_tokens,
            elapsed_seconds=elapsed_seconds,
            steps_per_second=steps_per_second,
            tokens_per_second=tokens_per_second,
            val_loss=val_loss,
            val_bpc=val_bpc,
        )

        # Track best validation
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_val_bpc = val_bpc
            self.best_val_step = step

        # Write to JSONL file
        self.metrics_file.write(json.dumps(metrics.to_dict()) + '\n')
        self.metrics_file.flush()

        return metrics

    def finalize(
        self,
        total_steps: int,
        total_time_seconds: float,
        final_val_loss: float,
    ) -> TrainingInfo:
        """Finalize training and save summary info.

        Args:
            total_steps: Total number of training steps
            total_time_seconds: Total training time
            final_val_loss: Final validation loss (can be JAX array or float)

        Returns:
            TrainingInfo object
        """
        # Close metrics file
        self.metrics_file.close()

        # Compute BPC and convert to Python types
        final_val_bpc = final_val_loss / jnp.log(2)
        final_val_bpc = float(final_val_bpc)

        # Create training info summary
        info = TrainingInfo(
            model_params=int(self.model_params),
            d_model=int(self.d_model),
            num_layers=int(self.num_layers),
            num_heads=int(self.num_heads),
            batch_size=int(self.batch_size),
            seq_len=int(self.seq_len),
            total_steps=int(total_steps),
            total_tokens=int(self.total_tokens),
            total_time_seconds=float(total_time_seconds),
            best_val_loss=float(self.best_val_loss),
            best_val_bpc=float(self.best_val_bpc),
            best_val_step=int(self.best_val_step),
            final_val_loss=final_val_loss,
            final_val_bpc=final_val_bpc,
            metrics_file=str(self.metrics_path.name),
        )

        # Save training info
        info.save_json(self.output_dir / "training-info.json")

        return info

    def __del__(self):
        """Ensure metrics file is closed."""
        if hasattr(self, 'metrics_file') and not self.metrics_file.closed:
            self.metrics_file.close()


def load_metrics(metrics_dir: Path | str) -> list[StepMetrics]:
    """Load all step metrics from a training run.

    Args:
        metrics_dir: Directory containing training-metrics.jsonl

    Returns:
        List of StepMetrics objects
    """
    metrics_path = Path(metrics_dir) / "training-metrics.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            metrics.append(StepMetrics(**data))

    return metrics


def load_training_info(metrics_dir: Path | str) -> TrainingInfo:
    """Load training info summary.

    Args:
        metrics_dir: Directory containing training-info.json

    Returns:
        TrainingInfo object
    """
    info_path = Path(metrics_dir) / "training-info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Training info not found: {info_path}")

    return TrainingInfo.load_json(info_path)


def print_training_summary(metrics_dir: Path | str) -> None:
    """Print a summary of training metrics.

    Args:
        metrics_dir: Directory containing training metrics
    """
    info = load_training_info(metrics_dir)
    metrics_dir = Path(metrics_dir).resolve()

    print(f"Training Summary (from '{metrics_dir}'):\n")
    print(f"Model:")
    print(f"  Parameters:    {info.model_params:,}")
    print(f"  d_model:       {info.d_model}")
    print(f"  Layers:        {info.num_layers}")
    print(f"  Heads:         {info.num_heads}")

    print(f"\nTraining:")
    print(f"  Total steps:   {info.total_steps:,}")
    print(f"  Total tokens:  {info.total_tokens:,} ({info.total_tokens / 1e6:.2f}M)")
    print(f"  Batch size:    {info.batch_size}")
    print(f"  Seq length:    {info.seq_len}")
    print(f"  Time:          {info.total_time_seconds / 60:.2f} minutes")

    print(f"\nBest Validation:")
    print(f"  Loss:          {info.best_val_loss:.4f}")
    print(f"  BPC:           {info.best_val_bpc:.4f}")
    print(f"  Step:          {info.best_val_step:,}")

    print(f"\nFinal Validation:")
    print(f"  Loss:          {info.final_val_loss:.4f}")
    print(f"  BPC:           {info.final_val_bpc:.4f}")

    # Compute tokens per second
    tokens_per_second = info.total_tokens / info.total_time_seconds
    print(f"\nThroughput:")
    print(f"  Tokens/sec:    {tokens_per_second:,.0f}")
    print(f"  Steps/sec:     {info.total_steps / info.total_time_seconds:.2f}")

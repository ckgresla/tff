"""Configuration objects for models and training using Pydantic."""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for GPT model architecture."""

    vocab_size: int = Field(default=256, description="Vocabulary size (256 for byte-level)")
    d_model: int = Field(default=512, description="Model embedding dimension")
    num_layers: int = Field(default=8, description="Number of transformer layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    d_ff: int = Field(default=2048, description="Feed-forward network dimension")
    max_seq_len: int = Field(default=512, description="Maximum sequence length")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout probability")

    class Config:
        frozen = True  # Make config immutable


class DataConfig(BaseModel):
    """Configuration for data loading."""

    data_path: str = Field(default="/raid/datasets/enwiki8.zip", description="Path to enwiki8.zip")
    seq_len: int = Field(default=256, description="Sequence length for training")
    batch_size: int = Field(default=32, description="Training batch size")

    class Config:
        frozen = True


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""

    learning_rate: float = Field(default=3e-4, gt=0.0, description="Learning rate")
    num_steps: int = Field(default=10000, gt=0, description="Total training steps")
    eval_every: int = Field(default=500, gt=0, description="Evaluate every N steps")
    log_every: int = Field(default=100, gt=0, description="Log every N steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Checkpointing
    checkpoint_dir: Optional[str] = Field(default=None, description="Directory to save checkpoints")
    save_every: int = Field(default=1000, gt=0, description="Save checkpoint every N steps")

    # Parallelism
    data_parallel: bool = Field(
        default=False,
        description="Enable data parallelism across all visible devices. "
        "Batch will be split evenly across devices. "
        "Use CUDA_VISIBLE_DEVICES to control which GPUs are used."
    )

    class Config:
        frozen = True


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration combining all config objects.

    # init with defaults
    `config = ExperimentConfig()`
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    class Config:
        frozen = True

    def save_json(self, path: Path | str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration.

        Returns:
            Multi-line string describing the configuration
        """
        return json.dumps(self.model_dump(), indent=2)

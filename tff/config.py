"""Configuration objects for models and training using Pydantic."""

import os
import json
from pathlib import Path
from typing import Optional, Literal, Union

from pydantic import BaseModel, Field, Discriminator
from typing_extensions import Annotated


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


# Optimizer configurations (discriminated union)
class AdamWConfig(BaseModel):
    """Configuration for AdamW optimizer."""

    name: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=3e-4, gt=0.0, description="Learning rate")
    beta1: float = Field(default=0.9, ge=0.0, le=1.0, description="Adam beta1 (first moment decay)")
    beta2: float = Field(default=0.999, ge=0.0, le=1.0, description="Adam beta2 (second moment decay)")
    eps: float = Field(default=1e-8, gt=0.0, description="Epsilon for numerical stability")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay coefficient")

    class Config:
        frozen = True


class AdamConfig(BaseModel):
    """Configuration for Adam optimizer."""

    name: Literal["adam"] = "adam"
    learning_rate: float = Field(default=3e-4, gt=0.0, description="Learning rate")
    beta1: float = Field(default=0.9, ge=0.0, le=1.0, description="Adam beta1 (first moment decay)")
    beta2: float = Field(default=0.999, ge=0.0, le=1.0, description="Adam beta2 (second moment decay)")
    eps: float = Field(default=1e-8, gt=0.0, description="Epsilon for numerical stability")

    class Config:
        frozen = True


class SGDConfig(BaseModel):
    """Configuration for SGD optimizer with optional momentum."""

    name: Literal["sgd"] = "sgd"
    learning_rate: float = Field(default=1e-2, gt=0.0, description="Learning rate")
    momentum: float = Field(default=0.0, ge=0.0, le=1.0, description="Momentum coefficient (0 for no momentum)")
    nesterov: bool = Field(default=False, description="Use Nesterov momentum")

    class Config:
        frozen = True


# Union of all optimizer configs with discriminator on 'name' field
OptimizerConfig = Annotated[
    Union[AdamWConfig, AdamConfig, SGDConfig],
    Discriminator('name')
]


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""

    optimizer: OptimizerConfig = Field(default_factory=AdamWConfig, description="Optimizer configuration")
    num_steps: int = Field(default=10000, gt=0, description="Total training steps")
    eval_every: int = Field(default=500, gt=0, description="Evaluate every N steps")
    eval_on_start: bool = Field(default=True, description="Evaluate model before training starts")
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

    # Weights & Biases logging (read from environment variables)
    wandb_project: Optional[str] = Field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT"),
        description="W&B project name from WANDB_PROJECT env var. If None, wandb logging is disabled."
    )
    wandb_entity: Optional[str] = Field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY"),
        description="W&B team/entity name from WANDB_ENTITY env var."
    )
    wandb_name: Optional[str] = Field(
        default_factory=lambda: os.environ.get("WANDB_NAME"),
        description="W&B run name from WANDB_NAME env var. If None, wandb generates a name."
    )
    wandb_tags: Optional[list[str]] = Field(
        default_factory=lambda: (
            os.environ.get("WANDB_TAGS").split(",")
            if os.environ.get("WANDB_TAGS")
            else None
        ),
        description="W&B tags from WANDB_TAGS env var (comma-separated, e.g., 'train,model-v2')."
    )
    wandb_run_url: Optional[str] = Field(
        default=None,
        description="W&B run URL (populated after wandb.init())."
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

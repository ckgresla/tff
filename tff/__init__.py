"""TFF - Transformer Former: Baseline and Dynamic Transformers for JAX."""

# Baseline GPT model
from tff.modeling import GPT

# Dynamic routing transformer (experimental)
from tff.model import DynamicTransformer
from tff.components import TransformerLayer, MultiHeadAttention, FeedForward
from tff.routing import LayerRouter, LayerPool

# Configuration
from tff.config import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
    AdamWConfig,
    AdamConfig,
    SGDConfig,
)

# Checkpoint utilities
from tff.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_best_checkpoint,
    load_final_checkpoint,
    load_step_checkpoint,
    list_checkpoints,
)

# Training
from tff.train import train

# Data loading
from tff.data import Enwik8Dataset

# Metrics
from tff.metrics import (
    MetricsTracker,
    load_metrics,
    load_training_info,
    print_training_summary,
    StepMetrics,
    TrainingInfo,
)

# Logging
from tff.logging import TrainingLogger

__all__ = [
    # Models
    "GPT",
    "DynamicTransformer",
    "TransformerLayer",
    "MultiHeadAttention",
    "FeedForward",
    "LayerRouter",
    "LayerPool",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "AdamWConfig",
    "AdamConfig",
    "SGDConfig",
    # Checkpoints
    "save_checkpoint",
    "load_checkpoint",
    "load_best_checkpoint",
    "load_final_checkpoint",
    "load_step_checkpoint",
    "list_checkpoints",
    # Training
    "train",
    # Data
    "Enwik8Dataset",
    # Metrics
    "MetricsTracker",
    "load_metrics",
    "load_training_info",
    "print_training_summary",
    "StepMetrics",
    "TrainingInfo",
    # Logging
    "TrainingLogger",
]

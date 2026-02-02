"""Configuration objects for models and training using Hydra structured configs."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hydra_zen import store
from omegaconf import MISSING, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for GPT model architecture."""

    vocab_size: int = 256
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout_rate: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_path: str = "/raid/datasets/enwiki8.zip"
    seq_len: int = 256
    batch_size: int = 32


@dataclass
class OptimizerConfig:
    """Base optimizer configuration. Subclassed by each optimizer variant."""

    name: str = "adamw"
    learning_rate: float = 3e-4


@dataclass
class AdamWConfig(OptimizerConfig):
    """Configuration for AdamW optimizer."""

    name: str = "adamw"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class AdamConfig(OptimizerConfig):
    """Configuration for Adam optimizer."""

    name: str = "adam"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SGDConfig(OptimizerConfig):
    """Configuration for SGD optimizer with optional momentum."""

    name: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.0
    nesterov: bool = False


_OPTIMIZER_REGISTRY: dict[str, type[OptimizerConfig]] = {
    "adamw": AdamWConfig,
    "adam": AdamConfig,
    "sgd": SGDConfig,
}


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    num_steps: int = 10000
    eval_every: int = 500
    eval_on_start: bool = True
    log_every: int = 100
    seed: int = 42

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_every: int = 1000

    # Parallelism
    data_parallel: bool = False

    # Weights & Biases logging (populated from env via OmegaConf oc.env resolver)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_run_url: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all config objects."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamWConfig)

    @classmethod
    def from_dictconfig(cls, cfg: "DictConfig") -> "ExperimentConfig":
        """Construct typed ExperimentConfig from a Hydra DictConfig.

        Resolves interpolations, picks the correct optimizer dataclass
        based on the 'name' field, and returns a fully typed instance.
        """
        import dataclasses as _dc

        d = OmegaConf.to_container(cfg, resolve=True)

        def _pick(dc_cls: type, section: dict) -> dict:
            """Keep only keys that are actual dataclass fields."""
            valid = {f.name for f in _dc.fields(dc_cls)}
            return {k: v for k, v in section.items() if k in valid}

        opt_dict = d["optimizer"]
        opt_cls = _OPTIMIZER_REGISTRY[opt_dict["name"]]

        return cls(
            model=ModelConfig(**_pick(ModelConfig, d["model"])),
            data=DataConfig(**_pick(DataConfig, d["data"])),
            training=TrainingConfig(**_pick(TrainingConfig, d["training"])),
            optimizer=opt_cls(**_pick(opt_cls, opt_dict)),
        )

    def save_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        container = OmegaConf.to_container(
            OmegaConf.structured(self), resolve=True
        )
        with open(path, "w") as f:
            json.dump(container, f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            config_dict = json.load(f)
        cfg = OmegaConf.structured(cls(**config_dict))
        return OmegaConf.to_object(cfg)

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        container = OmegaConf.to_container(
            OmegaConf.structured(self), resolve=True
        )
        return json.dumps(container, indent=2)


@dataclass
class _HydraExperimentConfig:
    """Hydra entry-point config with untyped group fields.

    Fields use ``Any = MISSING`` so hydra-zen's ``builds()`` wrapper and
    Hydra's defaults-list composition can populate them without type conflicts.
    The typed ``ExperimentConfig`` is reconstructed via ``from_dictconfig()``.
    """

    model: Any = MISSING
    data: Any = MISSING
    training: Any = MISSING
    optimizer: Any = MISSING


def _register_configs() -> None:
    """Register all structured configs with Hydra via hydra-zen store.

    All config groups (model, data, training, optimizer) and the top-level
    config are registered here — no YAML files needed.
    """
    # Model variants
    store(ModelConfig, group="model", name="default")
    store(ModelConfig, group="model", name="toy",
          d_model=128, num_layers=4, num_heads=4, d_ff=512, max_seq_len=256)

    # Data
    store(DataConfig, group="data", name="enwik8")

    # Training variants (env vars via OmegaConf resolver)
    _wandb_env = dict(
        wandb_project="${oc.env:WANDB_PROJECT,null}",
        wandb_entity="${oc.env:WANDB_ENTITY,null}",
        wandb_name="${oc.env:WANDB_NAME,null}",
        wandb_tags="${oc.env:WANDB_TAGS,null}",
    )
    store(TrainingConfig, group="training", name="default", **_wandb_env)
    store(TrainingConfig, group="training", name="toy",
          eval_every=1000, checkpoint_dir="checkpoints/toy", **_wandb_env)

    # Optimizers
    store(AdamWConfig, group="optimizer", name="adamw")
    store(AdamConfig, group="optimizer", name="adam")
    store(SGDConfig, group="optimizer", name="sgd")

    # Top-level config with defaults
    store(_HydraExperimentConfig, name="config",
          hydra_defaults=["_self_",
                          {"model": "default"},
                          {"data": "enwik8"},
                          {"training": "default"},
                          {"optimizer": "adamw"}])

    store.add_to_hydra_store()


_register_configs()

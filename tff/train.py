"""Clean training script for byte-level language modeling on enwik8.

A baseline training loop using JAX + Equinox with a simple GPT-style transformer.
Supports data parallelism across multiple GPUs.
"""

import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf

from tff.data import Enwik8Dataset
from tff.modeling import GPT
from tff.config import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
)
from tff.checkpoint import save_checkpoint
from tff.metrics import MetricsTracker
from tff.logging import setup_logging

import wandb

log = logging.getLogger("tff.train")


def create_optimizer(opt_config) -> optax.GradientTransformation:
    """Create optimizer from config.

    Args:
        opt_config: Optimizer configuration dataclass

    Returns:
        Optax optimizer
    """
    match opt_config.name:
        case "adamw":
            return optax.adamw(
                learning_rate=opt_config.learning_rate,
                b1=opt_config.beta1,
                b2=opt_config.beta2,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        case "adam":
            return optax.adam(
                learning_rate=opt_config.learning_rate,
                b1=opt_config.beta1,
                b2=opt_config.beta2,
                eps=opt_config.eps,
            )
        case "sgd":
            if opt_config.momentum > 0:
                return optax.sgd(
                    learning_rate=opt_config.learning_rate,
                    momentum=opt_config.momentum,
                    nesterov=opt_config.nesterov,
                )
            else:
                return optax.sgd(learning_rate=opt_config.learning_rate)
        case _:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")


def compute_update_norm(updates) -> float:
    """Compute global L2 norm of parameter updates."""
    update_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(u))
        for u in jax.tree_util.tree_leaves(updates)
        if u is not None
    ))
    return float(update_norm)


def compute_sample_loss(
    input_seq: Int[Array, "seq"],
    target_seq: Int[Array, "seq"],
    key,
    *,
    model: GPT,
):
    logits = model(input_seq, dropout_key=key)  # logits shape: (seq, vocab)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_seq)
    return jnp.mean(loss)  # Mean over sequence
#
@eqx.filter_jit
def compute_loss(
    model: GPT,
    inputs: Int[Array, "batch seq"],
    targets: Int[Array, "batch seq"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Compute loss for a batch using vmap.

    Model processes single sequences; we vmap over batch dimension.
    """
    batch_size = inputs.shape[0]
    keys = jr.split(key, batch_size)
    compute_sample_loss_with_model = partial(compute_sample_loss, model=model)
    losses = jax.vmap(compute_sample_loss_with_model)(inputs, targets, keys)  # (batch,)
    return jnp.mean(losses)



def make_train_step(model_sharding, data_sharding):
    """Create a train step function with sharding."""
    @eqx.filter_jit(donate="all")
    def train_step(
        model: GPT,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        inputs: Int[Array, "batch seq"],
        targets: Int[Array, "batch seq"],
        key: PRNGKeyArray,
    ) -> tuple[GPT, optax.OptState, Float[Array, ""], GPT, GPT]:
        """Single training step with sharding."""
        # Apply sharding constraints
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

        # Compute loss and gradients
        loss, grads = eqx.filter_value_and_grad(compute_loss)(
            model, inputs, targets, key
        )

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # Return with sharding
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        return model, opt_state, loss, grads, updates

    return train_step


def make_eval_step(model_sharding, data_sharding):
    """Create an eval step function with sharding."""
    @eqx.filter_jit
    def eval_step(
        model: GPT,
        inputs: Int[Array, "batch seq"],
        targets: Int[Array, "batch seq"],
    ) -> Float[Array, ""]:
        """Single evaluation step (no dropout)."""
        model = eqx.filter_shard(model, model_sharding)
        inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

        compute_sample_loss_with_model = partial(compute_sample_loss, model=model)
        loss_per_sample = jax.vmap(compute_sample_loss_with_model)(inputs, targets, None)  # (batch,)
        return jnp.mean(loss_per_sample)

    return eval_step


def train(config: ExperimentConfig) -> GPT:
    """Train a GPT model using a configuration object.

    Args:
        config: Experiment configuration containing model, data, training, and optimizer settings.

    Returns:
        Trained GPT model
    """
    m = config.model
    d = config.data
    t = config.training
    o = config.optimizer

    log.info("Training GPT on enwik8")
    cfg_yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
    log.info("Config:\n%s", cfg_yaml.rstrip())

    # Setup checkpoint directory
    checkpoint_path: Optional[Path] = None
    log_dir: Optional[Path] = None
    if t.checkpoint_dir is not None:
        checkpoint_path = Path(t.checkpoint_dir).resolve()
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        config_file = checkpoint_path / "experiment_config.json"
        config.save_json(config_file)
        log.info("Checkpoint: '%s'", checkpoint_path)

        log_dir = checkpoint_path / "logs"
    else:
        log_dir = Path("logs")

    # Re-initialize logging with file handler now that we know log_dir
    setup_logging(log_dir=log_dir)

    # JAX device info and parallelism setup
    devices = jax.devices()
    num_devices = len(devices)
    data_parallel = t.data_parallel

    # Setup sharding
    if data_parallel and num_devices > 1:
        if d.batch_size % num_devices != 0:
            raise ValueError(
                f"Batch size ({d.batch_size}) must be divisible by "
                f"number of devices ({num_devices}) for data parallelism. "
                f"Either adjust batch_size or set CUDA_VISIBLE_DEVICES."
            )
        per_device_batch_size = d.batch_size // num_devices
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
        log.info(
            "Devices: %dx %s | DP: True | Mesh: ('batch': %d) | Batch: %d (%d/dev)",
            num_devices, jax.default_backend(), num_devices, d.batch_size, per_device_batch_size,
        )
    else:
        mesh = jax.make_mesh((1,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
        log.info("Devices: %dx %s | DP: False", num_devices, jax.default_backend())

    # Load data
    log.info("Loading data...")
    train_data: Enwik8Dataset = Enwik8Dataset(
        d.data_path, seq_len=d.seq_len, split="train"
    )
    valid_data: Enwik8Dataset = Enwik8Dataset(
        d.data_path, seq_len=d.seq_len, split="valid"
    )

    # Initialize model
    log.info("Initializing model...")
    key: PRNGKeyArray = jr.PRNGKey(t.seed)
    model_key: PRNGKeyArray
    data_key: PRNGKeyArray
    model_key, data_key = jr.split(key)

    model: GPT = GPT(
        vocab_size=m.vocab_size,
        d_model=m.d_model,
        num_layers=m.num_layers,
        num_heads=m.num_heads,
        d_ff=m.d_ff,
        max_seq_len=m.max_seq_len,
        dropout_rate=m.dropout_rate,
        key=model_key,
    )

    num_params: int = model.count_parameters()
    log.info("Model: %s params | Optimizer: %s (lr=%s)", f"{num_params:,}", o.name, o.learning_rate)

    # Initialize optimizer
    optimizer: optax.GradientTransformation = create_optimizer(o)
    opt_state: optax.OptState = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Create train and eval step functions with appropriate sharding
    train_step_fn = make_train_step(model_sharding, data_sharding)
    eval_step_fn = make_eval_step(model_sharding, data_sharding)

    # Apply initial sharding
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    # Diagnostic: sharding info for multi-device
    if data_parallel and num_devices > 1:
        log.debug("=" * 70)
        log.debug("SHARDING DIAGNOSTICS")
        log.debug("=" * 70)
        log.debug("Sharding Configuration:")
        log.debug("  Model sharding: %s", model_sharding)
        log.debug("  Data sharding: %s", data_sharding)

        sample_param = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))[0]
        log.debug("Sample Parameter (wte.weight):")
        log.debug("  Shape: %s", sample_param.shape)
        log.debug("  Sharding: %s", sample_param.sharding)
        log.debug("  Is fully replicated: %s", sample_param.sharding.is_fully_replicated)

        if hasattr(sample_param, 'addressable_shards'):
            log.debug("  Per-device parameter info:")
            for device_idx, shard in enumerate(sample_param.addressable_shards):
                log.debug("    Device %d: shape=%s, index=%s", device_idx, shard.data.shape, shard.index)

            if len(sample_param.addressable_shards) > 1:
                first_val_dev0 = float(sample_param.addressable_shards[0].data.flatten()[0])
                first_val_dev1 = float(sample_param.addressable_shards[1].data.flatten()[0])
                is_replicated = abs(first_val_dev0 - first_val_dev1) < 1e-6
                log.debug("  Replication check (first parameter value):")
                log.debug("    Device 0: %.6f", first_val_dev0)
                log.debug("    Device 1: %.6f", first_val_dev1)
                log.debug("    Status: %s", "REPLICATED" if is_replicated else "NOT REPLICATED")

        log.debug("=" * 70)

    # Initialize wandb if enabled
    use_wandb = t.wandb_project is not None

    if use_wandb:
        wandb_tags = None
        if t.wandb_tags:
            wandb_tags = t.wandb_tags.split(",")

        wandb_config = OmegaConf.to_container(
            OmegaConf.structured(config), resolve=True
        )

        wandb.init(
            project=t.wandb_project,
            entity=t.wandb_entity,
            name=t.wandb_name,
            tags=wandb_tags,
            config=wandb_config,
        )

        wandb_run_url = wandb.run.get_url() if wandb.run else None
        wandb_parts = [f"WandB: {t.wandb_project}"]
        if t.wandb_name:
            wandb_parts.append(t.wandb_name)
        if wandb_run_url:
            wandb_parts.append(wandb_run_url)
        log.info(" | ".join(wandb_parts))

        # Re-save config with wandb URL
        if wandb_run_url and checkpoint_path is not None:
            config_file = checkpoint_path / "experiment_config.json"
            config.save_json(config_file)

    # Initialize metrics tracker
    metrics_tracker: Optional[MetricsTracker] = None
    if checkpoint_path is not None:
        metrics_tracker = MetricsTracker(
            output_dir=checkpoint_path,
            batch_size=d.batch_size,
            seq_len=d.seq_len,
            model_params=num_params,
            d_model=m.d_model,
            num_layers=m.num_layers,
            num_heads=m.num_heads,
        )

    # Training loop
    tokens_per_batch: int = d.batch_size * d.seq_len
    tokens_per_epoch: int = len(train_data.data)
    log.info(
        "Training: %d steps | %s tok/batch | %s tok/epoch",
        t.num_steps, f"{tokens_per_batch:,}", f"{tokens_per_epoch:,}",
    )

    train_key: PRNGKeyArray = data_key
    train_batches = train_data.iterate_batches(d.batch_size, train_key)

    best_val_loss: float = float('inf')
    start_time: float = time.time()

    # Evaluate before training starts (baseline random init performance)
    if t.eval_on_start:
        eval_key: PRNGKeyArray
        eval_key, _ = jr.split(train_key)
        eval_batches = valid_data.iterate_batches(
            d.batch_size, eval_key, num_batches=50
        )

        eval_losses: list[float] = []
        for eval_inputs, eval_targets in eval_batches:
            eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)
            eval_loss: Float[Array, ""] = eval_step_fn(model, eval_inputs, eval_targets)
            eval_losses.append(float(eval_loss))

        baseline_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
        baseline_bpc: Float[Array, ""] = baseline_loss / jnp.log(2)
        log.info("Baseline | Loss: %.4f | BPC: %.4f", float(baseline_loss), float(baseline_bpc))

        if use_wandb:
            wandb.log({
                "valid/loss": float(baseline_loss),
                "valid/bpc": float(baseline_bpc),
                "step": 0,
                "tokens": 0,
                "epoch": 0.0,
            })

    for step in range(t.num_steps):
        # Get batch
        inputs: Int[Array, "batch seq"]
        targets: Int[Array, "batch seq"]
        inputs, targets = next(train_batches)

        # Shard data
        inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

        # Diagnostic: batch shapes on first iteration
        if step == 0 and data_parallel and num_devices > 1:
            log.debug("=" * 70)
            log.debug("DATA PARALLEL VERIFICATION (Step 0)")
            log.debug("=" * 70)
            log.debug("Batch Information:")
            log.debug("  Global batch shape: %s", inputs.shape)
            log.debug("  Batch sharding: %s", inputs.sharding)
            log.debug("  Number of devices: %d", len(jax.devices()))

            log.debug("Per-device LOCAL data shapes:")
            for device_idx, device in enumerate(jax.devices()):
                local_shape = inputs.addressable_shards[device_idx].data.shape
                shard_index = inputs.addressable_shards[device_idx].index
                log.debug("  Device %d (%s:%s):", device_idx, device.device_kind, device.id)
                log.debug("    Local shape: %s", local_shape)
                log.debug("    Shard index: %s", shard_index)

            expected_per_device = inputs.shape[0] // len(jax.devices())
            actual_per_device = inputs.addressable_shards[0].data.shape[0]
            log.debug("Batch Split Verification:")
            log.debug("  Expected per-device batch size: %d", expected_per_device)
            log.debug("  Actual per-device batch size: %d", actual_per_device)
            if actual_per_device == expected_per_device:
                log.debug("  Status: CORRECTLY SPLIT across %d devices", len(jax.devices()))
            else:
                log.warning("  Status: NOT split as expected")

            log.debug("Data Uniqueness Check (first token on each device):")
            first_tokens = []
            for device_idx in range(min(len(jax.devices()), 4)):
                first_token = int(inputs.addressable_shards[device_idx].data[0, 0])
                first_tokens.append(first_token)
                log.debug("  Device %d: first token = %d", device_idx, first_token)

            if len(set(first_tokens)) > 1:
                log.debug("  Status: Devices have different data (expected for data parallel)")
            else:
                log.debug("  Note: All devices have same first token (could be coincidence)")

            log.debug("=" * 70)

        # Train step
        step_key: PRNGKeyArray
        train_key, step_key = jr.split(train_key)
        loss: Float[Array, ""]
        grads: GPT
        updates: GPT
        model, opt_state, loss, grads, updates = train_step_fn(
            model, opt_state, optimizer, inputs, targets, step_key
        )

        # Log training progress
        if step % t.log_every == 0 or step == t.num_steps - 1:
            elapsed: float = time.time() - start_time
            steps_per_sec: float = (step + 1) / elapsed if elapsed > 0 else 0
            bpc: float = float(loss / jnp.log(2))

            loss_py: float = float(loss)

            # Compute comprehensive metrics from model
            model_metrics = model.compute_metrics(grads)

            # Compute update norm
            update_norm = compute_update_norm(updates)

            # Log to metrics tracker
            if metrics_tracker is not None:
                metrics_tracker.log_step(
                    step=step,
                    loss=loss_py,
                    learning_rate=o.learning_rate,
                    elapsed_seconds=elapsed,
                )

            # Log core training metrics
            tokens_seen: int = (step + 1) * tokens_per_batch
            epoch = tokens_seen / tokens_per_epoch
            grad_norm = model_metrics['grad_norm/global']
            log.info(
                "Step %6d | epoch: %.2f | Loss: %.4f | BPC: %.4f | "
                "grad: %.2f | %.2f steps/s | %.2fM tok",
                step, epoch, loss_py, bpc, grad_norm,
                steps_per_sec, tokens_seen / 1e6,
            )

            # Log to wandb
            if use_wandb:
                tokens = (step + 1) * tokens_per_batch
                epoch = tokens / tokens_per_epoch
                wandb_metrics = {
                    "train/loss": loss_py,
                    "train/bpc": bpc,
                    "opt/step": step,
                    "opt/learning_rate": o.learning_rate,
                    "opt/update_norm": update_norm,
                    "throughput/steps_per_sec": steps_per_sec,
                    "throughput/tokens": tokens,
                    "step": step,
                    "tokens": tokens,
                    "epoch": epoch,
                }
                for metric_name, metric_value in model_metrics.items():
                    wandb_metrics[f"health/{metric_name}"] = metric_value

                wandb.log(wandb_metrics)

        # Evaluate
        if step % t.eval_every == 0 and step > 0:
            eval_key: PRNGKeyArray
            eval_key, _ = jr.split(train_key)
            eval_batches = valid_data.iterate_batches(
                d.batch_size, eval_key, num_batches=50
            )

            eval_losses: list[float] = []
            eval_inputs: Int[Array, "batch seq"]
            eval_targets: Int[Array, "batch seq"]
            for eval_inputs, eval_targets in eval_batches:
                eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)
                eval_loss: Float[Array, ""] = eval_step_fn(model, eval_inputs, eval_targets)
                eval_losses.append(float(eval_loss))

            avg_eval_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
            eval_bpc: Float[Array, ""] = avg_eval_loss / jnp.log(2)

            if metrics_tracker is not None:
                elapsed: float = time.time() - start_time
                metrics_tracker.log_step(
                    step=step,
                    loss=float(loss),
                    learning_rate=o.learning_rate,
                    elapsed_seconds=elapsed,
                    val_loss=float(avg_eval_loss),
                )

            if use_wandb:
                tokens = (step + 1) * tokens_per_batch
                epoch = tokens / tokens_per_epoch
                wandb.log({
                    "valid/loss": float(avg_eval_loss),
                    "valid/bpc": float(eval_bpc),
                    "throughput/tokens": tokens,
                    "step": step,
                    "tokens": tokens,
                    "epoch": epoch,
                })

            is_best = avg_eval_loss < best_val_loss
            if is_best:
                best_val_loss = avg_eval_loss
                if checkpoint_path is not None:
                    save_checkpoint(model, config, checkpoint_path, "best-model")
                log.info(
                    "Eval %6d | Loss: %.4f | BPC: %.4f | new best → saved",
                    step, float(avg_eval_loss), float(eval_bpc),
                )
            else:
                log.info(
                    "Eval %6d | Loss: %.4f | BPC: %.4f",
                    step, float(avg_eval_loss), float(eval_bpc),
                )

        # Save periodic checkpoint
        if checkpoint_path is not None and step % t.save_every == 0 and step > 0:
            ckpt_name = f"checkpoint-{step:06d}"
            save_checkpoint(model, config, checkpoint_path, ckpt_name)
            log.info("Checkpoint saved: '%s/%s.eqx'", checkpoint_path, ckpt_name)

    # Final evaluation
    eval_key, _ = jr.split(train_key)
    eval_batches = valid_data.iterate_batches(d.batch_size, eval_key, num_batches=100)

    eval_losses = []
    for eval_inputs, eval_targets in eval_batches:
        eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)
        eval_loss = eval_step_fn(model, eval_inputs, eval_targets)
        eval_losses.append(float(eval_loss))

    final_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
    final_bpc: Float[Array, ""] = final_loss / jnp.log(2)
    best_bpc: Float[Array, ""] = best_val_loss / jnp.log(2)

    # Save final model
    if checkpoint_path is not None:
        save_checkpoint(model, config, checkpoint_path, "final-model")

    total_time: float = time.time() - start_time
    log.info(
        "Done | Loss: %.4f | BPC: %.4f | Best BPC: %.4f | %.1fmin",
        float(final_loss), float(final_bpc), float(best_bpc), total_time / 60,
    )

    # Finalize metrics tracking
    if metrics_tracker is not None:
        metrics_tracker.finalize(
            total_steps=t.num_steps,
            total_time_seconds=total_time,
            final_val_loss=float(final_loss),
        )

    # Finalize wandb
    if use_wandb:
        wandb.finish()

    return model


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training (Hydra-powered)."""
    # Force GPU usage
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

    # Disable Hydra's root logger handlers to prevent duplicate output
    logging.getLogger().handlers.clear()
    # Suppress noisy JAX backend warnings (e.g. "Unable to initialize backend 'tpu'")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

    # Convert DictConfig -> typed dataclass
    config: ExperimentConfig = ExperimentConfig.from_dictconfig(cfg)

    # Setup logging (console-only initially; train() will add file handler)
    setup_logging()

    # Train model
    model: GPT = train(config)


if __name__ == "__main__":
    main()

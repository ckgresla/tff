"""Clean training script for byte-level language modeling on enwik8.

A baseline training loop using JAX + Equinox with a simple GPT-style transformer.
Supports data parallelism across multiple GPUs.
"""

import os
import time
import fire
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from tff.data import Enwik8Dataset
from tff.modeling import GPT
from tff.config import (
    ExperimentConfig, ModelConfig, DataConfig, TrainingConfig,
    OptimizerConfig, AdamWConfig, AdamConfig, SGDConfig
)
from tff.checkpoint import save_checkpoint
from tff.metrics import MetricsTracker, print_training_summary
from tff.logging import TrainingLogger

import wandb


def create_optimizer(opt_config: OptimizerConfig) -> optax.GradientTransformation:
    """Create optimizer from config.

    Args:
        opt_config: Optimizer configuration (discriminated union)

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
    """Compute global L2 norm of parameter updates.

    Args:
        updates: PyTree of parameter updates

    Returns:
        L2 norm of all updates
    """
    update_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(u))
        for u in jax.tree_util.tree_leaves(updates)
        if u is not None
    ))
    return float(update_norm)


@eqx.filter_jit
def compute_loss(
    model: GPT,
    inputs: Int[Array, "batch seq"],
    targets: Int[Array, "batch seq"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Compute loss for a batch using vmap.

    Model processes single sequences; we vmap over batch dimension.

    Args:
        model: GPT model (expects single sequence input)
        inputs: [batch, seq] - Input token indices
        targets: [batch, seq] - Target token indices
        key: PRNG key for dropout

    Returns:
        Scalar loss value
    """
    batch_size = inputs.shape[0]

    # Split key for each sequence in batch
    keys = jr.split(key, batch_size)

    # Vmap model over batch dimension
    # Lambda to handle keyword-only argument
    logits = jax.vmap(lambda idx, k: model(idx, key=k), in_axes=(0, 0))(inputs, keys)  # (batch, seq, vocab)

    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,  # (batch, seq, vocab)
        labels=targets,  # (batch, seq)
    )
    return jnp.mean(loss)


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
        """Single training step with sharding.

        Args:
            model: GPT model
            opt_state: Optimizer state
            optimizer: Optax optimizer
            inputs: [batch, seq] - Input token indices
            targets: [batch, seq] - Target token indices
            key: PRNG key for dropout

        Returns:
            Tuple of (updated model, updated optimizer state, loss value, gradients, updates)
        """
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
        """Single evaluation step (no dropout).

        Args:
            model: GPT model (expects single sequence input)
            inputs: [batch, seq] - Input token indices
            targets: [batch, seq] - Target token indices

        Returns:
            Scalar loss value
        """
        # Apply sharding constraints
        model = eqx.filter_shard(model, model_sharding)
        inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

        # Vmap model over batch dimension (no dropout keys)
        logits = jax.vmap(lambda x: model(x, key=None))(inputs)  # (batch, seq, vocab)

        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=targets,
        )
        return jnp.mean(loss)

    return eval_step


def train(config: ExperimentConfig) -> GPT:
    """Train a GPT model using a Pydantic configuration object.

    Args:
        config: Experiment configuration containing model, data, and training settings.
                Set config.training.data_parallel=True for multi-GPU data parallelism.

    Returns:
        Trained GPT model

    Example:
        >>> from tff.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
        >>> from tff.train import train
        >>>
        >>> # Create custom configuration
        >>> config = ExperimentConfig(
        ...     model=ModelConfig(d_model=512, num_layers=8),
        ...     data=DataConfig(batch_size=32),
        ...     training=TrainingConfig(num_steps=10000, checkpoint_dir="checkpoints", data_parallel=True),
        ... )
        >>>
        >>> # Train with data parallelism
        >>> model = train(config)
    """
    print("Training GPT on enwik8\n")
    print(config.summary())

    # Setup checkpoint directory
    checkpoint_path: Optional[Path] = None
    log_dir: Optional[Path] = None
    if config.training.checkpoint_dir is not None:
        checkpoint_path = Path(config.training.checkpoint_dir).resolve()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        print(f"\nCheckpoint directory: '{checkpoint_path}'")

        # Save initial config
        config_file = checkpoint_path / "experiment_config.json"
        config.save_json(config_file)
        print(f"Saved config to: '{config_file}'")

        # Setup log directory (separate from config)
        log_dir = checkpoint_path / "logs"
    else:
        # If no checkpoint dir, still log to logs/ in current directory
        log_dir = Path("logs")

    # Initialize training logger (always enabled)
    log_dir.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir)
    print(f"Training logs: '{log_dir.resolve()}'")

    # Log configuration
    training_logger.log_config({
        "model": {
            "vocab_size": config.model.vocab_size,
            "d_model": config.model.d_model,
            "num_layers": config.model.num_layers,
            "num_heads": config.model.num_heads,
            "d_ff": config.model.d_ff,
            "max_seq_len": config.model.max_seq_len,
            "dropout_rate": config.model.dropout_rate,
        },
        "data": {
            "data_path": config.data.data_path,
            "seq_len": config.data.seq_len,
            "batch_size": config.data.batch_size,
        },
        "training": {
            "learning_rate": config.training.optimizer.learning_rate,
            "num_steps": config.training.num_steps,
            "eval_every": config.training.eval_every,
            "eval_on_start": config.training.eval_on_start,
            "log_every": config.training.log_every,
            "seed": config.training.seed,
            "data_parallel": config.training.data_parallel,
        }
    })

    # JAX device info and parallelism setup
    devices = jax.devices()
    num_devices = len(devices)
    data_parallel = config.training.data_parallel

    print(f"\nJAX devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Num devices: {num_devices}")

    # Setup sharding - always create shardings (single-device when not parallel)
    print(f"\nData parallelism: {data_parallel}")
    print(f"  Devices: {num_devices}")
    print(f"  Global batch size: {config.data.batch_size}")
    if data_parallel and num_devices > 1:
        # Validate batch size is divisible by num_devices
        if config.data.batch_size % num_devices != 0:
            raise ValueError(
                f"Batch size ({config.data.batch_size}) must be divisible by "
                f"number of devices ({num_devices}) for data parallelism. "
                f"Either adjust batch_size or set CUDA_VISIBLE_DEVICES."
            )
        per_device_batch_size = config.data.batch_size // num_devices
        print(f"  Per-device batch size: {per_device_batch_size}")
        # Create mesh and shardings for multi-device
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())  # Replicated

    else:
        # Single-device sharding (effectively a no-op)
        mesh = jax.make_mesh((1,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    print(f"[ MESH ]\n{mesh}")

    # Load the actual data
    # TODO: make this more generic, so can do other tasks and stuff
    data_path = Path(config.data.data_path).resolve()
    print(f"\nLoading data from '{data_path}'...")
    train_data: Enwik8Dataset = Enwik8Dataset(
        config.data.data_path, seq_len=config.data.seq_len, split="train"
    )
    valid_data: Enwik8Dataset = Enwik8Dataset(
        config.data.data_path, seq_len=config.data.seq_len, split="valid"
    )

    # Initialize model
    print("\nInitializing model...")
    key: PRNGKeyArray = jr.PRNGKey(config.training.seed)
    model_key: PRNGKeyArray
    data_key: PRNGKeyArray
    model_key, data_key = jr.split(key)

    model: GPT = GPT(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout_rate=config.model.dropout_rate,
        key=model_key,
    )

    num_params: int = model.count_parameters()
    print(f"Model initialized with {num_params:,} parameters")

    # Initialize optimizer
    print(f"\nSetting up optimizer ({config.training.optimizer.name}, lr={config.training.optimizer.learning_rate})...")
    optimizer: optax.GradientTransformation = create_optimizer(config.training.optimizer)
    opt_state: optax.OptState = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Create train and eval step functions with appropriate sharding
    train_step_fn = make_train_step(model_sharding, data_sharding)
    eval_step_fn = make_eval_step(model_sharding, data_sharding)

    # Apply initial sharding to model and optimizer state (always, even single-device)
    print("\nApplying sharding to model and optimizer state...")
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    # Diagnostic: print shapes and sharding info
    if data_parallel and num_devices > 1:
        print("\n" + "="*70)
        print("SHARDING DIAGNOSTICS")
        print("="*70)
        print(f"\nSharding Configuration:")
        print(f"  Model sharding: {model_sharding}")
        print(f"  Data sharding: {data_sharding}")

        # Check a sample parameter's sharding
        sample_param = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))[0]
        print(f"\nSample Parameter (wte.weight):")
        print(f"  Shape: {sample_param.shape}")
        print(f"  Sharding: {sample_param.sharding}")
        print(f"  Is fully replicated: {sample_param.sharding.is_fully_replicated}")

        # Show per-device info for parameters
        if hasattr(sample_param, 'addressable_shards'):
            print(f"\n  Per-device parameter info:")
            for device_idx, shard in enumerate(sample_param.addressable_shards):
                print(f"    Device {device_idx}: shape={shard.data.shape}, index={shard.index}")

            # Verify replication by checking if first element is same on all devices
            if len(sample_param.addressable_shards) > 1:
                # Get values from different devices and convert to Python floats (moves to host)
                first_val_dev0 = float(sample_param.addressable_shards[0].data.flatten()[0])
                first_val_dev1 = float(sample_param.addressable_shards[1].data.flatten()[0])
                is_replicated = abs(first_val_dev0 - first_val_dev1) < 1e-6
                print(f"\n  Replication check (first parameter value):")
                print(f"    Device 0: {first_val_dev0:.6f}")
                print(f"    Device 1: {first_val_dev1:.6f}")
                print(f"    Status: {'REPLICATED' if is_replicated else 'NOT REPLICATED'}")

        print("="*70 + "\n")

    # Initialize wandb if enabled
    use_wandb = config.training.wandb_project is not None

    if use_wandb:
        print(f"\nInitializing Weights & Biases...")
        print(f"  Project: {config.training.wandb_project}")
        if config.training.wandb_entity:
            print(f"  Entity: {config.training.wandb_entity}")
        if config.training.wandb_name:
            print(f"  Run name: {config.training.wandb_name}")
        if config.training.wandb_tags:
            print(f"  Tags: {config.training.wandb_tags}")

        # Initialize wandb
        wandb.init(
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            name=config.training.wandb_name,
            tags=config.training.wandb_tags,
            config=config.model_dump(),  # Log full config
        )

        # Update config with wandb run URL
        wandb_run_url = wandb.run.get_url() if wandb.run else None
        if wandb_run_url:
            print(f"  Run URL: {wandb_run_url}")
            # Create updated config with wandb URL
            config = config.model_copy(
                update={"training": config.training.model_copy(update={"wandb_run_url": wandb_run_url})}
            )

            # Re-save config with wandb URL
            if checkpoint_path is not None:
                config_file = checkpoint_path / "experiment_config.json"
                config.save_json(config_file)
                print(f"  Updated config with run URL: '{config_file}'")

    # Initialize metrics tracker
    metrics_tracker: Optional[MetricsTracker] = None
    if checkpoint_path is not None:
        metrics_tracker = MetricsTracker(
            output_dir=checkpoint_path,
            batch_size=config.data.batch_size,
            seq_len=config.data.seq_len,
            model_params=num_params,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
        )
        metrics_file = checkpoint_path / "training-metrics.jsonl"
        print(f"Metrics will be saved to: '{metrics_file}'")

    # Training loop
    print("\nStarting training...")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Sequence length: {config.data.seq_len}")
    print(f"  Total steps: {config.training.num_steps}")
    tokens_per_batch: int = config.data.batch_size * config.data.seq_len
    print(f"  Tokens per batch: {tokens_per_batch:,}\n")

    train_key: PRNGKeyArray = data_key
    train_batches = train_data.iterate_batches(config.data.batch_size, train_key)

    best_val_loss: float = float('inf')
    start_time: float = time.time()

    # Evaluate before training starts (baseline random init performance)
    if config.training.eval_on_start:
        print("Evaluating baseline model (random initialization)...")

        eval_key: PRNGKeyArray
        eval_key, _ = jr.split(train_key)
        eval_batches = valid_data.iterate_batches(
            config.data.batch_size, eval_key, num_batches=50
        )

        eval_losses: list[float] = []
        for eval_inputs, eval_targets in eval_batches:
            eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)
            eval_loss: Float[Array, ""] = eval_step_fn(model, eval_inputs, eval_targets)
            eval_losses.append(float(eval_loss))

        baseline_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
        baseline_bpc: Float[Array, ""] = baseline_loss / jnp.log(2)

        print(f"  Baseline Loss: {float(baseline_loss):.4f}")
        print(f"  Baseline BPC:  {float(baseline_bpc):.4f}\n")

        training_logger.log_baseline_eval(float(baseline_loss), float(baseline_bpc))

        if use_wandb:
            wandb.log({
                "valid/loss": float(baseline_loss),
                "valid/bpc": float(baseline_bpc),
                "step": 0,
                "tokens": 0,
            })

    # Log training start
    training_logger.log_training_start()

    for step in range(config.training.num_steps):
        # Get batch
        inputs: Int[Array, "batch seq"]
        targets: Int[Array, "batch seq"]
        inputs, targets = next(train_batches)

        # Shard data (always - no-op for single device)
        inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

        # Diagnostic: print batch shapes on first iteration
        if step == 0 and data_parallel and num_devices > 1:
            print("\n" + "="*70)
            print("DATA PARALLEL VERIFICATION (Step 0)")
            print("="*70)
            print(f"\nBatch Information:")
            print(f"  Global batch shape: {inputs.shape}")
            print(f"  Batch sharding: {inputs.sharding}")
            print(f"  Number of devices: {len(jax.devices())}")

            # Show per-device LOCAL shapes
            print(f"\nPer-device LOCAL data shapes:")
            for device_idx, device in enumerate(jax.devices()):
                # Get the local slice on each device
                local_shape = inputs.addressable_shards[device_idx].data.shape
                shard_index = inputs.addressable_shards[device_idx].index
                print(f"  Device {device_idx} ({device.device_kind}:{device.id}):")
                print(f"    Local shape: {local_shape}")
                print(f"    Shard index: {shard_index}")

            # Verify the split
            expected_per_device = inputs.shape[0] // len(jax.devices())
            actual_per_device = inputs.addressable_shards[0].data.shape[0]
            print(f"\nBatch Split Verification:")
            print(f"  Expected per-device batch size: {expected_per_device}")
            print(f"  Actual per-device batch size: {actual_per_device}")
            if actual_per_device == expected_per_device:
                print(f"  Status: CORRECTLY SPLIT across {len(jax.devices())} devices")
            else:
                print(f"  Status: WARNING - NOT split as expected")

            # Show first element on each device to prove they have different data
            print(f"\nData Uniqueness Check (first token on each device):")
            first_tokens = []
            for device_idx in range(min(len(jax.devices()), 4)):  # Show up to 4 devices
                first_token = int(inputs.addressable_shards[device_idx].data[0, 0])
                first_tokens.append(first_token)
                print(f"  Device {device_idx}: first token = {first_token}")

            if len(set(first_tokens)) > 1:
                print(f"  Status: Devices have different data (expected for data parallel)")
            else:
                print(f"  Note: All devices have same first token (could be coincidence)")

            print("="*70 + "\n")

        # Train step
        step_key: PRNGKeyArray
        train_key, step_key = jr.split(train_key)
        loss: Float[Array, ""]
        grads: GPT
        updates: GPT
        model, opt_state, loss, grads, updates = train_step_fn(
            model, opt_state, optimizer, inputs, targets, step_key
        )

        # Log training progress (every log_every steps, plus the final step)
        if step % config.training.log_every == 0 or step == config.training.num_steps - 1:
            elapsed: float = time.time() - start_time
            steps_per_sec: float = (step + 1) / elapsed if elapsed > 0 else 0
            bpc: float = float(loss / jnp.log(2))

            # Convert JAX arrays to Python types at boundary
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
                    learning_rate=config.training.optimizer.learning_rate,
                    elapsed_seconds=elapsed,
                )

            # Print to console and log
            tokens_seen: int = (step + 1) * tokens_per_batch
            print(
                f"Step {step:6d} | "
                f"Loss: {loss_py:.4f} | "
                f"BPC: {bpc:.4f} | "
                f"grad_norm: {model_metrics['grad_norm/global']:.2f} | "
                f"{steps_per_sec:.2f} steps/s | "
                f"{tokens_seen / 1e6:.2f}M tokens"
            )

            # Log core training metrics
            training_logger.log_step(
                step=step,
                loss=loss_py,
                bpc=bpc,
                steps_per_sec=steps_per_sec,
                tokens_seen=tokens_seen,
            )

            # Log health metrics separately
            training_logger.log_health(step, model_metrics)

            # Log to wandb with organized namespaces
            if use_wandb:
                tokens = (step + 1) * tokens_per_batch
                wandb_metrics = {
                    # Core training metrics
                    "train/loss": loss_py,
                    "train/bpc": bpc,
                    # Optimizer state
                    "opt/step": step,
                    "opt/learning_rate": config.training.optimizer.learning_rate,
                    "opt/update_norm": update_norm,
                    # Throughput metrics (hardware/system performance)
                    "throughput/steps_per_sec": steps_per_sec,
                    "throughput/tokens": tokens,
                    # X-axis variables for wandb
                    "step": step,
                    "tokens": tokens,
                }
                # Health metrics (model diagnostics)
                for metric_name, metric_value in model_metrics.items():
                    wandb_metrics[f"health/{metric_name}"] = metric_value

                wandb.log(wandb_metrics)

        # Evaluate
        if step % config.training.eval_every == 0 and step > 0:
            print(f"\nValidation (step {step}):")

            # Eval on validation set
            eval_key: PRNGKeyArray
            eval_key, _ = jr.split(train_key)
            eval_batches = valid_data.iterate_batches(
                config.data.batch_size, eval_key, num_batches=50
            )

            eval_losses: list[float] = []
            eval_inputs: Int[Array, "batch seq"]
            eval_targets: Int[Array, "batch seq"]
            for eval_inputs, eval_targets in eval_batches:
                # Shard data (always - no-op for single device)
                eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)

                eval_loss: Float[Array, ""] = eval_step_fn(model, eval_inputs, eval_targets)
                eval_losses.append(float(eval_loss))

            avg_eval_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
            eval_bpc: Float[Array, ""] = avg_eval_loss / jnp.log(2)

            print(f"  Loss: {float(avg_eval_loss):.4f}")
            print(f"  BPC:  {float(eval_bpc):.4f}")

            if metrics_tracker is not None:
                elapsed: float = time.time() - start_time
                metrics_tracker.log_step(
                    step=step,
                    loss=float(loss),
                    learning_rate=config.training.optimizer.learning_rate,
                    elapsed_seconds=elapsed,
                    val_loss=float(avg_eval_loss),
                )

            if use_wandb:
                tokens = (step + 1) * tokens_per_batch
                wandb.log({
                    "valid/loss": float(avg_eval_loss),
                    "valid/bpc": float(eval_bpc),
                    "throughput/tokens": tokens,
                    "step": step,
                    "tokens": tokens,
                })

            is_best = avg_eval_loss < best_val_loss
            if is_best:
                best_val_loss = avg_eval_loss
                print(f"  New best! Saving checkpoint...")

                if checkpoint_path is not None:
                    save_checkpoint(model, config, checkpoint_path, "best-model")
                    best_model_path = checkpoint_path / "best-model.eqx"
                    print(f"  Saved to: '{best_model_path}'")

            training_logger.log_eval(step, float(avg_eval_loss), float(eval_bpc), is_best=bool(is_best))

        # Save periodic checkpoint
        if checkpoint_path is not None and step % config.training.save_every == 0 and step > 0:
            ckpt_name = f"checkpoint-{step:06d}"
            save_checkpoint(model, config, checkpoint_path, ckpt_name)
            ckpt_file = checkpoint_path / f"{ckpt_name}.eqx"
            print(f"\nSaved periodic checkpoint: '{ckpt_file}'")
            training_logger.log_checkpoint(step, "periodic")

    # Final evaluation
    print("\nTraining complete! Running final evaluation...")

    eval_key, _ = jr.split(train_key)
    eval_batches = valid_data.iterate_batches(config.data.batch_size, eval_key, num_batches=100)

    eval_losses = []
    for eval_inputs, eval_targets in eval_batches:
        # Shard data (always - no-op for single device)
        eval_inputs, eval_targets = eqx.filter_shard((eval_inputs, eval_targets), data_sharding)

        eval_loss = eval_step_fn(model, eval_inputs, eval_targets)
        eval_losses.append(float(eval_loss))

    final_loss: Float[Array, ""] = jnp.mean(jnp.array(eval_losses))
    final_bpc: Float[Array, ""] = final_loss / jnp.log(2)
    best_bpc: Float[Array, ""] = best_val_loss / jnp.log(2)

    print(f"\nFinal Validation:")
    print(f"  Loss:      {float(final_loss):.4f}")
    print(f"  BPC:       {float(final_bpc):.4f}")
    print(f"  Best BPC:  {float(best_bpc):.4f}")

    # Save final model
    if checkpoint_path is not None:
        save_checkpoint(model, config, checkpoint_path, "final-model")
        final_model_path = checkpoint_path / "final-model.eqx"
        print(f"\nSaved final checkpoint: '{final_model_path}'")
        training_logger.log_checkpoint(config.training.num_steps, "final")

    total_time: float = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.2f} minutes")

    training_logger.log_training_complete(
        total_steps=config.training.num_steps,
        total_time_minutes=total_time / 60,
        best_loss=float(best_val_loss),
        best_bpc=float(best_bpc),
        best_step=metrics_tracker.best_val_step if metrics_tracker else 0,
        final_loss=float(final_loss),
        final_bpc=float(final_bpc),
    )

    # Finalize metrics tracking
    if metrics_tracker is not None:
        training_info = metrics_tracker.finalize(
            total_steps=config.training.num_steps,
            total_time_seconds=total_time,
            final_val_loss=float(final_loss),
        )
        metrics_jsonl = checkpoint_path / "training-metrics.jsonl"
        info_json = checkpoint_path / "training-info.json"
        print(f"\nMetrics saved:")
        print(f"  '{metrics_jsonl}'")
        print(f"  '{info_json}'")

    # Finalize wandb
    if use_wandb:
        wandb.finish()
        print("\nWandB run finalized")
        if config.training.wandb_run_url:
            print(f"  Run URL: {config.training.wandb_run_url}")

    # Close training logger
    training_logger.close()
    print(f"\nTraining logs written to: '{log_dir.resolve()}'")

    return model



def main(data_parallel: bool = False) -> None:
    """Main entry point for training.

    Args:
        data_parallel: Enable data parallelism across all available GPUs.
                      Use CUDA_VISIBLE_DEVICES to control which GPUs are used.
                      Example: CUDA_VISIBLE_DEVICES=0,1 python -m tff.train --data_parallel
    """
    # Force GPU usage
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')

    # Create configuration
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            d_model=512,
            num_layers=8,
            num_heads=8,
            d_ff=2048,
            max_seq_len=512,
            dropout_rate=0.1,
        ),
        data=DataConfig(
            data_path="/raid/datasets/enwiki8.zip",
            seq_len=256,
            batch_size=32,
        ),
        training=TrainingConfig(
            learning_rate=3e-4,
            num_steps=10000,
            eval_every=500,
            log_every=100,
            seed=42,
            checkpoint_dir="checkpoints",
            save_every=1000,
            data_parallel=data_parallel,
        ),
    )

    # Train model
    model: GPT = train(config)


if __name__ == "__main__":
    fire.Fire(main)

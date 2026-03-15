"""Looped Transformer (Universal Transformer variant) in JAX + Equinox.

A single transformer block applied T times with sinusoidal coordinate
embeddings, following Dehghani et al., "Universal Transformers" (2018).
https://arxiv.org/abs/1807.03819

Official reference impl (TensorFlow/Tensor2Tensor):
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer.py

Deviations from the paper (intentional, for fair comparison with our baselines):
  - Pre-norm instead of post-norm (matches our RegularTransformer baseline)
  - GELU instead of ReLU in FFN (same reason)
  These ensure the only variable across models is the architecture
  (shared+looped vs distinct+sequential vs routed), not the norm/activation.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray

from choosy.modeling.transformer import Block


def _sinusoidal_embeddings(num_positions: int, d_model: int) -> Float[Array, "pos d"]:
    """Sinusoidal embeddings (same formula as Vaswani et al. positional encodings).

    Used for both position and timestep embeddings in the UT paper.
    """
    pos = jnp.arange(num_positions, dtype=jnp.float32)
    dim = jnp.arange(0, d_model, 2, dtype=jnp.float32)
    angles = pos[:, None] / jnp.power(10000.0, dim[None, :] / d_model)
    emb = jnp.zeros((num_positions, d_model))
    emb = emb.at[:, 0::2].set(jnp.sin(angles))
    emb = emb.at[:, 1::2].set(jnp.cos(angles))
    return emb


def _coordinate_embeddings(
    max_seq_len: int, num_steps: int, d_model: int,
) -> Float[Array, "steps seq d"]:
    """Coordinate embeddings combining position and timestep (UT paper eqs 6-7).

    P^t_{i,2j} = sin(i / 10000^{2j/d}) + sin(t / 10000^{2j/d})
    P^t_{i,2j+1} = cos(i / 10000^{2j/d}) + cos(t / 10000^{2j/d})

    Returns (num_steps, max_seq_len, d_model) — one embedding per (step, position).
    """
    pos_emb = _sinusoidal_embeddings(max_seq_len, d_model)   # (seq, d)
    step_emb = _sinusoidal_embeddings(num_steps, d_model)     # (steps, d)
    # Broadcast add: (steps, 1, d) + (1, seq, d) → (steps, seq, d)
    return step_emb[:, None, :] + pos_emb[None, :, :]


class LoopedTransformer(eqx.Module):
    """Looped Transformer for byte-level language modeling.

    A single transformer block with shared weights applied T times.
    Timestep embeddings tell the model which iteration it's on.

    Processes single sequences (T,) → (T, vocab_size).
    Batching happens via vmap at training loop level.
    """

    vocab_size: int
    d_model: int
    num_loops: int
    num_heads: int
    max_seq_len: int

    wte: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    block: Block  # single shared block
    ln_f: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear
    coord_emb: Array  # (num_loops, max_seq_len, d_model), frozen sinusoidal

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_loops: int = 8,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_loops = num_loops
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        keys = jr.split(key, 3)

        self.wte = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.drop = eqx.nn.Dropout(dropout_rate)

        # Single shared block (the core of the looped transformer)
        self.block = Block(d_model, num_heads, d_ff, max_seq_len, dropout_rate, key=keys[1])

        self.ln_f = eqx.nn.LayerNorm(d_model)
        self.lm_head = eqx.nn.Linear(d_model, vocab_size, key=keys[2])

        # Coordinate embeddings: position + timestep (per UT paper eqs 6-7, not trained)
        self.coord_emb = _coordinate_embeddings(max_seq_len, num_loops, d_model)

    def __call__(
        self,
        idx: Int[Array, "seq"],
        *,
        dropout_key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq vocab"]:
        T, = idx.shape

        # Token embeddings only (position info comes from coordinate embeddings each iteration)
        x = jax.vmap(self.wte)(idx)

        # Dropout
        if dropout_key is not None:
            drop_key, *loop_keys = jr.split(dropout_key, self.num_loops + 1)
            x = self.drop(x, key=drop_key)
        else:
            loop_keys = [None] * self.num_loops

        # Apply the SAME block T times with coordinate embeddings (UT paper)
        # Each iteration re-injects both position and timestep information
        for t in range(self.num_loops):
            x = x + self.coord_emb[t, :T, :]  # (T, d_model)
            x = self.block(x, dropout_key=loop_keys[t])

        # Final layer norm + projection
        x = jax.vmap(self.ln_f)(x)
        logits = jax.vmap(self.lm_head)(x)
        return logits

    def count_parameters(self) -> int:
        params = eqx.filter(self, eqx.is_inexact_array)
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    def compute_metrics(self, grads: "LoopedTransformer") -> dict[str, float]:
        metrics = {}

        # Global gradient norm
        grad_params = eqx.filter(grads, eqx.is_inexact_array)
        grad_leaves = jax.tree_util.tree_leaves(grad_params)
        global_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))
        metrics["grad_norm/global"] = float(global_grad_norm)

        # Embeddings
        wte_grads = eqx.filter(grads.wte, eqx.is_inexact_array)
        metrics["grad_norm/wte"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(wte_grads))))

        # Shared block (gradients accumulate across loop iterations)
        block_params = eqx.filter(grads.block, eqx.is_inexact_array)
        metrics["grad_norm/shared_block"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(block_params))))

        attn_grads = eqx.filter(grads.block.attn, eqx.is_inexact_array)
        metrics["grad_norm/shared_block_attn"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(attn_grads))))

        mlp_grads = eqx.filter(grads.block.mlp, eqx.is_inexact_array)
        metrics["grad_norm/shared_block_mlp"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(mlp_grads))))

        # Head
        ln_f_grads = eqx.filter(grads.ln_f, eqx.is_inexact_array)
        lm_head_grads = eqx.filter(grads.lm_head, eqx.is_inexact_array)
        metrics["grad_norm/ln_f"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(ln_f_grads))))
        metrics["grad_norm/lm_head"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(lm_head_grads))))

        # Global parameter norm
        model_params = eqx.filter(self, eqx.is_inexact_array)
        metrics["param_norm/global"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(model_params))))

        return metrics

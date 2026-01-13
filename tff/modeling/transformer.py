"""Baseline GPT-style Transformer implementation with JAX + Equinox.

A clean, nanoGPT-inspired implementation following JAX best practices.
Model processes single sequences; batching happens via vmap at training level.
"""

import math
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Optional


class CausalSelfAttention(eqx.Module):
    """Causal multi-head self-attention.

    Processes a single sequence (T, d_model).
    """

    num_heads: int
    d_model: int
    head_dim: int
    c_attn: eqx.nn.Linear  # Combined Q, K, V projection
    c_proj: eqx.nn.Linear  # Output projection
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    bias: Array  # Causal mask (static, not trainable)

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        key1, key2 = jr.split(key)

        # Combined QKV projection (like nanoGPT)
        self.c_attn = eqx.nn.Linear(d_model, 3 * d_model, key=key1)
        self.c_proj = eqx.nn.Linear(d_model, d_model, key=key2)

        self.attn_dropout = eqx.nn.Dropout(dropout_rate)
        self.resid_dropout = eqx.nn.Dropout(dropout_rate)

        # Causal mask (not trainable)
        self.bias = jnp.tril(jnp.ones((max_seq_len, max_seq_len)))

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq d_model"]:
        T, C = x.shape

        # QKV projection - single vmap over sequence dimension
        qkv = jax.vmap(self.c_attn)(x)  # (T, 3*C)
        q, k, v = jnp.split(qkv, 3, axis=1)  # Each (T, C)

        # Reshape for multi-head attention
        k = k.reshape(T, self.num_heads, self.head_dim).swapaxes(0, 1)  # (nh, T, hs)
        q = q.reshape(T, self.num_heads, self.head_dim).swapaxes(0, 1)  # (nh, T, hs)
        v = v.reshape(T, self.num_heads, self.head_dim).swapaxes(0, 1)  # (nh, T, hs)

        # Scaled dot-product attention
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        scores = jnp.where(self.bias[:T, :T] == 0, float('-inf'), scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply dropout to attention weights
        if key is not None:
            key1, key2 = jr.split(key)
            attn_weights = self.attn_dropout(attn_weights, key=key1)
        else:
            key2 = None

        # Apply attention to values
        y = jnp.matmul(attn_weights, v)  # (nh, T, hs)

        # Reshape back
        y = y.swapaxes(0, 1).reshape(T, C)  # (T, C)

        # Output projection
        y = jax.vmap(self.c_proj)(y)

        # Residual dropout
        if key2 is not None:
            y = self.resid_dropout(y, key=key2)

        return y


class MLP(eqx.Module):
    """Position-wise feed-forward network.

    Processes a single sequence (T, d_model).
    """

    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jr.split(key)
        self.c_fc = eqx.nn.Linear(d_model, d_ff, key=key1)
        self.c_proj = eqx.nn.Linear(d_ff, d_model, key=key2)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq d_model"]:
        # Vectorized over sequence dimension
        x = jax.vmap(self.c_fc)(x)
        x = jax.nn.gelu(x, approximate=True)  # Use approximate GELU like nanoGPT

        if key is not None:
            key1, key2 = jr.split(key)
            x = self.dropout(x, key=key1)
        else:
            key2 = None

        x = jax.vmap(self.c_proj)(x)

        if key2 is not None:
            x = self.dropout(x, key=key2)

        return x


class Block(eqx.Module):
    """Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP.

    Processes a single sequence (T, d_model).
    """

    ln_1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jr.split(key)

        self.ln_1 = eqx.nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout_rate, key=key1)
        self.ln_2 = eqx.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout_rate, key=key2)

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq d_model"]:
        # Prepare keys
        if key is not None:
            key1, key2 = jr.split(key)
        else:
            key1 = key2 = None

        # Pre-norm architecture
        x = x + self.attn(jax.vmap(self.ln_1)(x), key=key1)
        x = x + self.mlp(jax.vmap(self.ln_2)(x), key=key2)

        return x


class GPT(eqx.Module):
    """GPT-style transformer for byte-level language modeling.

    Processes single sequences (T,) → (T, vocab_size).
    Batching happens via vmap at training loop level.
    """

    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    max_seq_len: int

    wte: eqx.nn.Embedding  # Token embeddings
    wpe: eqx.nn.Embedding  # Position embeddings
    drop: eqx.nn.Dropout
    blocks: list[Block]
    ln_f: eqx.nn.LayerNorm  # Final layer norm
    lm_head: eqx.nn.Linear  # Language model head

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary (256 for byte-level)
            d_model: Model dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension (typically 4 * d_model)
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            key: PRNG key for initialization
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Split keys for different components
        keys = jr.split(key, num_layers + 4)

        # Token embedding
        self.wte = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])

        # Position embedding
        self.wpe = eqx.nn.Embedding(max_seq_len, d_model, key=keys[1])

        # Dropout
        self.drop = eqx.nn.Dropout(dropout_rate)

        # Transformer blocks
        self.blocks = [
            Block(d_model, num_heads, d_ff, max_seq_len, dropout_rate, key=keys[i + 2])
            for i in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = eqx.nn.LayerNorm(d_model)

        # Language model head (projects to vocabulary)
        self.lm_head = eqx.nn.Linear(d_model, vocab_size, key=keys[-1])

    def __call__(
        self,
        idx: Int[Array, "seq"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq vocab"]:
        """
        Forward pass for a single sequence.

        Args:
            idx: Input token IDs [seq] (single sequence, no batch dim)
            key: PRNG key for dropout

        Returns:
            logits: Output logits [seq, vocab]
        """
        T, = idx.shape

        # Position indices
        pos = jnp.arange(0, T, dtype=jnp.int32)

        # Token + position embeddings (vmap over sequence)
        tok_emb = jax.vmap(self.wte)(idx)  # (T, C)
        pos_emb = jax.vmap(self.wpe)(pos)  # (T, C)
        x = tok_emb + pos_emb

        # Dropout
        if key is not None:
            drop_key, *block_keys = jr.split(key, self.num_layers + 1)
            x = self.drop(x, key=drop_key)
        else:
            block_keys = [None] * self.num_layers

        # Apply transformer blocks
        for block, block_key in zip(self.blocks, block_keys):
            x = block(x, key=block_key)

        # Final layer norm
        x = jax.vmap(self.ln_f)(x)

        # Project to vocabulary
        logits = jax.vmap(self.lm_head)(x)  # (T, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        params = eqx.filter(self, eqx.is_inexact_array)
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    def compute_metrics(self, grads: "GPT") -> dict[str, float]:
        """Compute comprehensive training metrics.

        Args:
            grads: Gradient tree (same structure as model)

        Returns:
            Dictionary of metrics with string keys and float values
        """
        metrics = {}

        # === Global gradient norm ===
        grad_params = eqx.filter(grads, eqx.is_inexact_array)
        grad_leaves = jax.tree_util.tree_leaves(grad_params)
        global_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))
        metrics["grad_norm/global"] = float(global_grad_norm)

        # === Per-layer gradient norms ===
        # Embeddings
        wte_grads = eqx.filter(grads.wte, eqx.is_inexact_array)
        wpe_grads = eqx.filter(grads.wpe, eqx.is_inexact_array)
        metrics["grad_norm/wte"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(wte_grads))))
        metrics["grad_norm/wpe"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(wpe_grads))))

        # Per transformer block
        for i, block_grad in enumerate(grads.blocks):
            block_params = eqx.filter(block_grad, eqx.is_inexact_array)
            block_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(block_params)))
            metrics[f"grad_norm/block_{i}"] = float(block_norm)

            # Attention subcomponents
            attn_grads = eqx.filter(block_grad.attn, eqx.is_inexact_array)
            attn_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(attn_grads)))
            metrics[f"grad_norm/block_{i}_attn"] = float(attn_norm)

            # MLP subcomponents
            mlp_grads = eqx.filter(block_grad.mlp, eqx.is_inexact_array)
            mlp_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(mlp_grads)))
            metrics[f"grad_norm/block_{i}_mlp"] = float(mlp_norm)

        # Final layer norm and head
        ln_f_grads = eqx.filter(grads.ln_f, eqx.is_inexact_array)
        lm_head_grads = eqx.filter(grads.lm_head, eqx.is_inexact_array)
        metrics["grad_norm/ln_f"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(ln_f_grads))))
        metrics["grad_norm/lm_head"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(lm_head_grads))))

        # === Global parameter norm ===
        model_params = eqx.filter(self, eqx.is_inexact_array)
        model_leaves = jax.tree_util.tree_leaves(model_params)
        global_param_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in model_leaves))
        metrics["param_norm/global"] = float(global_param_norm)

        # === Per-layer parameter norms ===
        # Embeddings
        wte_params = eqx.filter(self.wte, eqx.is_inexact_array)
        wpe_params = eqx.filter(self.wpe, eqx.is_inexact_array)
        metrics["param_norm/wte"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(wte_params))))
        metrics["param_norm/wpe"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(wpe_params))))

        # Per transformer block
        for i, block in enumerate(self.blocks):
            block_params = eqx.filter(block, eqx.is_inexact_array)
            block_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(block_params)))
            metrics[f"param_norm/block_{i}"] = float(block_norm)

        # Final layer norm and head
        ln_f_params = eqx.filter(self.ln_f, eqx.is_inexact_array)
        lm_head_params = eqx.filter(self.lm_head, eqx.is_inexact_array)
        metrics["param_norm/ln_f"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(ln_f_params))))
        metrics["param_norm/lm_head"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(lm_head_params))))

        return metrics

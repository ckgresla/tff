"""Core transformer building blocks using Equinox."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional


class MultiHeadAttention(eqx.Module):
    """Multi-head self-attention mechanism."""

    num_heads: int
    d_model: int
    head_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, num_heads: int, d_model: int, dropout_rate: float = 0.1, *, key: PRNGKeyArray):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        keys = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=keys[0])
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=keys[1])
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=keys[2])
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=keys[3])
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        mask: Optional[Float[Array, "batch heads seq seq"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "batch seq d_model"]:
        batch_size, seq_len, _ = x.shape

        # Linear projections - reshape to (batch*seq, d_model), apply, reshape back
        x_flat = x.reshape(-1, self.d_model)
        q = jax.lax.map(self.q_proj, x_flat).reshape(batch_size, seq_len, self.d_model)
        k = jax.lax.map(self.k_proj, x_flat).reshape(batch_size, seq_len, self.d_model)
        v = jax.lax.map(self.v_proj, x_flat).reshape(batch_size, seq_len, self.d_model)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attention_weights = jax.nn.softmax(scores, axis=-1)

        if key is not None:
            attention_weights = self.dropout(attention_weights, key=key)

        # Apply attention to values
        attended = jnp.matmul(attention_weights, v)

        # Reshape back
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.d_model)

        # Final linear projection
        attended_flat = attended.reshape(-1, self.d_model)
        output = jax.lax.map(self.out_proj, attended_flat).reshape(batch_size, seq_len, self.d_model)

        return output


class FeedForward(eqx.Module):
    """Position-wise feed-forward network."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, *, key: PRNGKeyArray):
        keys = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(d_model, d_ff, key=keys[0])
        self.fc2 = eqx.nn.Linear(d_ff, d_model, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "batch seq d_model"]:
        batch_size, seq_len, d_model = x.shape

        # Flatten, apply, reshape
        x_flat = x.reshape(-1, d_model)
        x = jax.lax.map(self.fc1, x_flat)
        x = jax.nn.gelu(x)

        if key is not None:
            key1, key2 = jr.split(key)
            x = self.dropout(x, key=key1)

        x = jax.lax.map(self.fc2, x)

        if key is not None:
            x = self.dropout(x, key=key2)

        # Reshape back
        return x.reshape(batch_size, seq_len, d_model)


class TransformerLayer(eqx.Module):
    """A single transformer layer with attention and feed-forward."""

    attention: MultiHeadAttention
    ffn: FeedForward
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, 2)
        self.attention = MultiHeadAttention(num_heads, d_model, dropout_rate, key=keys[0])
        self.ffn = FeedForward(d_model, d_ff, dropout_rate, key=keys[1])
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        mask: Optional[Float[Array, "batch heads seq seq"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "batch seq d_model"]:
        # Self-attention with residual connection and layer norm
        if key is not None:
            key1, key2 = jr.split(key)
        else:
            key1 = key2 = None

        batch_size, seq_len, d_model = x.shape

        attended = self.attention(x, mask=mask, key=key1)
        residual = (x + attended).reshape(-1, d_model)
        x = jax.lax.map(self.ln1, residual).reshape(batch_size, seq_len, d_model)

        # Feed-forward with residual connection and layer norm
        ff_output = self.ffn(x, key=key2)
        residual2 = (x + ff_output).reshape(-1, d_model)
        x = jax.lax.map(self.ln2, residual2).reshape(batch_size, seq_len, d_model)

        return x

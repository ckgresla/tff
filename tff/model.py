"""Dynamic Routing Transformer Model."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

from tff.routing import LayerRouter, LayerPool


class DynamicTransformer(eqx.Module):
    """
    A transformer that dynamically routes through a pool of layers.

    Instead of applying layers in a fixed sequence, this model:
    1. Maintains a pool of N transformer layers
    2. At each computation step, routes the current representation to one of the layers
    3. Applies the selected layer and repeats for K steps

    This allows the model to learn adaptive computation paths through the layer pool.
    """

    d_model: int
    vocab_size: int
    num_steps: int
    embedding: eqx.nn.Embedding
    layer_pool: LayerPool
    router: LayerRouter
    output_projection: eqx.nn.Linear
    pos_embedding: Float[Array, "max_seq d_model"]
    max_seq_len: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_pool_layers: int,
        num_steps: int,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        router_hidden_size: int = 256,
        router_temperature: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize the dynamic routing transformer.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_pool_layers: Number of layers in the pool
            num_steps: Number of routing steps to perform
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            router_hidden_size: Hidden size of the router MLP
            router_temperature: Temperature for routing softmax
            key: PRNG key for initialization
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.max_seq_len = max_seq_len

        keys = jr.split(key, 5)

        # Token embedding
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])

        # Learnable positional embeddings
        self.pos_embedding = jr.normal(keys[1], (max_seq_len, d_model)) * 0.02

        # Layer pool
        self.layer_pool = LayerPool(
            num_layers=num_pool_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            key=keys[2],
        )

        # Router
        self.router = LayerRouter(
            d_model=d_model,
            num_layers=num_pool_layers,
            hidden_size=router_hidden_size,
            temperature=router_temperature,
            key=keys[3],
        )

        # Output projection to vocabulary
        self.output_projection = eqx.nn.Linear(d_model, vocab_size, key=keys[4])

    def __call__(
        self,
        tokens: Float[Array, "batch seq"],
        mask: Optional[Float[Array, "batch heads seq seq"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        training: bool = False,
    ) -> tuple[Float[Array, "batch seq vocab"], dict]:
        """
        Forward pass with dynamic routing.

        Args:
            tokens: Input token IDs [batch, seq]
            mask: Attention mask
            key: PRNG key for dropout and routing
            training: Whether we're in training mode

        Returns:
            logits: Output logits [batch, seq, vocab]
            info: Dictionary containing routing information and statistics
        """
        batch_size, seq_len = tokens.shape

        # Embed tokens - flatten, embed, reshape
        tokens_flat = tokens.reshape(-1)
        x = jax.lax.map(self.embedding, tokens_flat).reshape(batch_size, seq_len, self.d_model)

        # Add positional embeddings
        x = x + self.pos_embedding[:seq_len]

        # Track routing decisions across steps
        all_layer_choices = []
        all_router_logits = []

        # Prepare keys for each step
        if key is not None:
            step_keys = jr.split(key, self.num_steps)
        else:
            step_keys = [None] * self.num_steps

        # Dynamic routing loop
        for step in range(self.num_steps):
            step_key = step_keys[step]

            if step_key is not None:
                router_key, layer_key = jr.split(step_key)
            else:
                router_key = layer_key = None

            # Route: decide which layer to apply
            layer_indices, router_logits = self.router(
                x, key=router_key, training=training
            )

            # Apply the selected layers
            x = self.layer_pool.apply_routed(
                layer_indices, x, mask=mask, key=layer_key
            )

            # Track routing decisions
            all_layer_choices.append(layer_indices)
            all_router_logits.append(router_logits)

        # Project to vocabulary - flatten, project, reshape
        x_flat = x.reshape(-1, self.d_model)
        logits = jax.lax.map(self.output_projection, x_flat).reshape(batch_size, seq_len, self.vocab_size)

        # Compile routing information
        info = {
            "layer_choices": jnp.stack(all_layer_choices, axis=1),  # [batch, num_steps]
            "router_logits": jnp.stack(all_router_logits, axis=1),  # [batch, num_steps, num_layers]
        }

        return logits, info

    def compute_routing_loss(
        self,
        router_logits: Float[Array, "batch num_steps num_layers"],
        balance_weight: float = 0.01,
    ) -> Float[Array, ""]:
        """
        Compute auxiliary loss to encourage balanced layer usage.

        Args:
            router_logits: Router logits from forward pass
            balance_weight: Weight for the balance loss

        Returns:
            Auxiliary loss encouraging uniform layer usage
        """
        # Compute average routing probabilities across batch and steps
        probs = jax.nn.softmax(router_logits, axis=-1)
        avg_probs = jnp.mean(probs, axis=(0, 1))  # [num_layers]

        # Encourage uniform distribution
        num_layers = avg_probs.shape[0]
        target_prob = 1.0 / num_layers
        balance_loss = jnp.sum((avg_probs - target_prob) ** 2)

        return balance_weight * balance_loss

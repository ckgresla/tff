"""Dynamic layer routing mechanisms."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Optional

from tff.components import TransformerLayer


class LayerRouter(eqx.Module):
    """Routes to different layers based on current representation.

    This router computes logits for each layer in the pool and selects
    the next layer to apply via softmax (during training) or argmax (inference).
    """

    d_model: int
    num_layers: int
    router_net: eqx.nn.MLP
    temperature: float

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        hidden_size: int = 256,
        temperature: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.temperature = temperature

        # MLP that maps from [d_model] -> [num_layers]
        # Takes the mean-pooled representation and outputs routing logits
        self.router_net = eqx.nn.MLP(
            in_size=d_model,
            out_size=num_layers,
            width_size=hidden_size,
            depth=2,
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        *,
        key: Optional[PRNGKeyArray] = None,
        training: bool = False,
    ) -> tuple[Int[Array, "batch"], Float[Array, "batch num_layers"]]:
        """
        Compute routing decision for each batch element.

        Args:
            x: Input representation [batch, seq, d_model]
            key: PRNG key for sampling during training
            training: Whether we're in training mode (sample) or inference (argmax)

        Returns:
            layer_indices: Selected layer for each batch element [batch]
            router_logits: Routing logits for all layers [batch, num_layers]
        """
        # Mean pool over sequence dimension to get a summary vector
        x_pooled = jnp.mean(x, axis=1)  # [batch, d_model]

        # Get routing logits - MLP already handles the last dimension correctly
        # so we just need to map over the batch dimension
        logits = jax.lax.map(self.router_net, x_pooled)  # [batch, num_layers]
        logits = logits / self.temperature

        if training and key is not None:
            # Sample from the distribution during training
            layer_indices = jax.random.categorical(key, logits, axis=-1)
        else:
            # Argmax during inference
            layer_indices = jnp.argmax(logits, axis=-1)

        return layer_indices, logits


class LayerPool(eqx.Module):
    """A pool of transformer layers that can be applied dynamically."""

    layers: list[TransformerLayer]
    num_layers: int

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        self.num_layers = num_layers
        keys = jr.split(key, num_layers)

        self.layers = [
            TransformerLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                key=k,
            )
            for k in keys
        ]

    def _apply_single(
        self,
        layer_idx: int,
        x: Float[Array, "seq d_model"],
        mask: Optional[Float[Array, "heads seq seq"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq d_model"]:
        """Apply a layer to a single sequence using lax.switch."""
        # Create a list of functions, one for each layer
        def make_layer_fn(layer):
            def fn(args):
                x, mask, key = args
                # Add batch dimension for layer
                x_batch = x[None, ...]
                mask_batch = mask[None, ...] if mask is not None else None
                out = layer(x_batch, mask=mask_batch, key=key)
                # Remove batch dimension
                return out[0]
            return fn

        layer_fns = [make_layer_fn(layer) for layer in self.layers]

        # Use lax.switch to dynamically select and apply the layer
        return jax.lax.switch(layer_idx, layer_fns, (x, mask, key))

    def apply_routed(
        self,
        layer_indices: Int[Array, "batch"],
        x: Float[Array, "batch seq d_model"],
        mask: Optional[Float[Array, "batch heads seq seq"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "batch seq d_model"]:
        """
        Apply different layers to different batch elements based on routing.

        Uses jax.lax.switch for efficient dynamic dispatch that works on GPU.

        Args:
            layer_indices: Which layer to apply for each batch element [batch]
            x: Input tensor [batch, seq, d_model]
            mask: Attention mask
            key: PRNG key

        Returns:
            Output tensor [batch, seq, d_model] where each batch element has been
            processed by its corresponding selected layer.
        """
        batch_size = x.shape[0]

        # Generate keys for each batch element if needed
        if key is not None:
            keys = jr.split(key, batch_size)
        else:
            keys = [None] * batch_size

        # Map over batch dimension - use manual loop since lax.map doesn't support multiple args well
        outputs = []
        for i in range(batch_size):
            out = self._apply_single(
                layer_indices[i],
                x[i],
                mask[i] if mask is not None else None,
                key=keys[i]
            )
            outputs.append(out)

        return jnp.stack(outputs)

"""ChoosyTransformer: dynamic layer routing from a pool, in JAX + Equinox.

Instead of applying layers in fixed sequential order (RegularTransformer) or
looping a single shared block (LoopedTransformer), the ChoosyTransformer
maintains a pool of N distinct transformer blocks and a router that selects
which block to apply at each of K routing steps.

Routing uses Gumbel-softmax with straight-through estimation:
  - Forward: hard routing (argmax), selected layer output at full strength
  - Backward: soft gradients flow to the router via the ST estimator
  - No output dilution, router learns selection preferences from main loss

Block pool uses stacked parameters (all N blocks' weights in single arrays
with leading N dim) + dynamic indexing instead of lax.switch. This gives
O(1) JIT compilation regardless of pool size.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray

from choosy.modeling.transformer import Block
from choosy.modeling.looped import _sinusoidal_embeddings


def _stack_blocks(blocks: list[Block]) -> Block:
    """Stack N blocks into a single Block-shaped pytree with leading N dim.

    Array leaves get shape (N, ...), non-array leaves keep the first block's value.
    """
    return jax.tree.map(
        lambda *leaves: jnp.stack(leaves) if eqx.is_array(leaves[0]) else leaves[0],
        *blocks,
    )


def _select_block(pool: Block, idx) -> Block:
    """Index into a stacked pool to get one block's parameters.

    Array leaves: (N, ...) → (...) via [idx] indexing.
    Non-array leaves: passed through unchanged.
    """
    return jax.tree.map(
        lambda leaf: leaf[idx] if eqx.is_array(leaf) else leaf,
        pool,
    )


class ChoosyTransformer(eqx.Module):
    """ChoosyTransformer for byte-level language modeling.

    Pool of N transformer blocks with a learned router that selects
    which block to apply at each of K routing steps.

    Processes single sequences (T,) → (T, vocab_size).
    Returns (logits, router_logits) where router_logits has shape (K, N)
    for computing auxiliary losses and routing analysis.
    Batching happens via vmap at training loop level.
    """

    vocab_size: int
    d_model: int
    num_pool_layers: int   # N: number of distinct blocks in the pool
    num_routing_steps: int # K: number of times we route (independent of N)
    num_heads: int
    max_seq_len: int
    routing_loss_weight: float
    router_temperature: float

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    pool: Block              # stacked Block: array leaves have shape (N, ...)
    router: eqx.nn.MLP      # d_model → num_pool_layers
    content_gru: eqx.nn.GRUCell  # recurrent content vector for routing context
    step_emb: Array          # (num_routing_steps, d_model), sinusoidal
    ln_f: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_pool_layers: int = 8,
        num_routing_steps: int = 8,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        router_hidden_size: int = 256,
        router_temperature: float = 1.0,
        routing_loss_weight: float = 0.01,
        *,
        key: PRNGKeyArray,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_pool_layers = num_pool_layers
        self.num_routing_steps = num_routing_steps
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.routing_loss_weight = routing_loss_weight
        # Store as jnp scalar so it can be updated via eqx.tree_at for annealing
        self.router_temperature = jnp.float32(router_temperature)

        keys = jr.split(key, num_pool_layers + 5)

        self.wte = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.wpe = eqx.nn.Embedding(max_seq_len, d_model, key=keys[1])
        self.drop = eqx.nn.Dropout(dropout_rate)

        # Pool of N distinct transformer blocks, stacked for O(1) JIT
        blocks = [
            Block(d_model, num_heads, d_ff, max_seq_len, dropout_rate, key=keys[i + 2])
            for i in range(num_pool_layers)
        ]
        self.pool = _stack_blocks(blocks)

        # Router: content vector + step embedding → logits over N layers
        self.router = eqx.nn.MLP(
            in_size=d_model,
            out_size=num_pool_layers,
            width_size=router_hidden_size,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[num_pool_layers + 2],
        )

        # Recurrent content vector: GRU accumulates routing context across steps
        # Input: mean-pooled hidden state (d_model), State: content vector (d_model)
        self.content_gru = eqx.nn.GRUCell(d_model, d_model, key=keys[num_pool_layers + 3])

        # Step embeddings: tell the router which step it's on (frozen sinusoidal)
        self.step_emb = _sinusoidal_embeddings(num_routing_steps, d_model)

        self.ln_f = eqx.nn.LayerNorm(d_model)
        self.lm_head = eqx.nn.Linear(d_model, vocab_size, key=keys[num_pool_layers + 4])

    def _route_and_apply(
        self,
        x: Float[Array, "seq d_model"],
        content: Float[Array, "d_model"],
        step: int,
        router_key: PRNGKeyArray | None,
        block_key: PRNGKeyArray | None,
    ) -> tuple[Float[Array, "seq d_model"], Float[Array, "d_model"], Float[Array, "pool"]]:
        """Single routing step: router picks a layer, that layer is applied.

        Uses Gumbel-softmax ST during training (router_key is not None):
          - Forward: hard argmax selection, output at full strength
          - Backward: soft gradients via straight-through estimator

        During inference (router_key is None):
          - Deterministic argmax routing, no gating

        Returns:
            x_out: updated hidden state
            content_out: updated content vector (for next step)
            logits: router logits for this step
        """
        # Update content vector with current hidden state summary
        summary = x.mean(axis=0)                                    # (d_model,)
        content = self.content_gru(summary, content)                # (d_model,)

        # Router decision: content vector + step embedding
        logits = self.router(content + self.step_emb[step])         # (num_pool_layers,)

        if router_key is not None:
            # --- Training: Gumbel-softmax with straight-through ---
            gumbel_noise = -jnp.log(-jnp.log(
                jr.uniform(router_key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
            ))
            perturbed = (logits + gumbel_noise) / self.router_temperature
            soft_weights = jax.nn.softmax(perturbed)

            # Hard selection in forward pass
            selected = jnp.argmax(perturbed)

            # Index into stacked pool to get one block (no lax.switch!)
            block = _select_block(self.pool, selected)
            block_output = block(x, dropout_key=block_key)

            # Straight-through gate for router gradient flow
            hard_gate = jnp.float32(1.0)
            soft_gate = soft_weights[selected]
            gate = hard_gate + (soft_gate - jax.lax.stop_gradient(soft_gate))

            x_out = x + gate * (block_output - x)
        else:
            # --- Inference: deterministic argmax ---
            selected = jnp.argmax(logits)
            block = _select_block(self.pool, selected)
            x_out = block(x, dropout_key=None)

        return x_out, content, logits

    def __call__(
        self,
        idx: Int[Array, "seq"],
        *,
        dropout_key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "seq vocab"], Float[Array, "steps pool"]]:
        """Forward pass with dynamic routing.

        Returns:
            logits: (seq, vocab_size)
            router_logits: (num_routing_steps, num_pool_layers) for aux loss / analysis
        """
        T, = idx.shape
        pos = jnp.arange(T, dtype=jnp.int32)

        # Token + position embeddings
        x = jax.vmap(self.wte)(idx) + jax.vmap(self.wpe)(pos)

        # Dropout + key management
        if dropout_key is not None:
            drop_key, *step_keys = jr.split(dropout_key, self.num_routing_steps + 1)
            x = self.drop(x, key=drop_key)
        else:
            step_keys = [None] * self.num_routing_steps

        all_router_logits = []

        # Initialize recurrent content vector (zeros — GRU will populate)
        content = jnp.zeros(self.d_model)

        for step in range(self.num_routing_steps):
            if step_keys[step] is not None:
                router_key, block_key = jr.split(step_keys[step])
            else:
                router_key = block_key = None

            x, content, step_logits = self._route_and_apply(x, content, step, router_key, block_key)
            all_router_logits.append(step_logits)

        # Final layer norm + projection
        x = jax.vmap(self.ln_f)(x)
        output_logits = jax.vmap(self.lm_head)(x)

        router_logits = jnp.stack(all_router_logits)  # (K, N)
        return output_logits, router_logits

    def count_parameters(self) -> int:
        params = eqx.filter(self, eqx.is_inexact_array)
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    def compute_metrics(self, grads: "ChoosyTransformer") -> dict[str, float]:
        metrics = {}

        # Global gradient norm
        grad_params = eqx.filter(grads, eqx.is_inexact_array)
        grad_leaves = jax.tree_util.tree_leaves(grad_params)
        global_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))
        metrics["grad_norm/global"] = float(global_grad_norm)

        # Embeddings
        wte_grads = eqx.filter(grads.wte, eqx.is_inexact_array)
        wpe_grads = eqx.filter(grads.wpe, eqx.is_inexact_array)
        metrics["grad_norm/wte"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(wte_grads))))
        metrics["grad_norm/wpe"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(wpe_grads))))

        # Per-block gradient norms (from stacked pool grads)
        for i in range(self.num_pool_layers):
            block_grad = _select_block(grads.pool, i)
            block_params = eqx.filter(block_grad, eqx.is_inexact_array)
            block_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(block_params)))
            metrics[f"grad_norm/pool_block_{i}"] = float(block_norm)

        # Router gradient norm
        router_grads = eqx.filter(grads.router, eqx.is_inexact_array)
        metrics["grad_norm/router"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(router_grads))))

        # Head
        ln_f_grads = eqx.filter(grads.ln_f, eqx.is_inexact_array)
        lm_head_grads = eqx.filter(grads.lm_head, eqx.is_inexact_array)
        metrics["grad_norm/ln_f"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(ln_f_grads))))
        metrics["grad_norm/lm_head"] = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(lm_head_grads))))

        # Global parameter norm
        model_params = eqx.filter(self, eqx.is_inexact_array)
        metrics["param_norm/global"] = float(jnp.sqrt(sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(model_params))))

        return metrics

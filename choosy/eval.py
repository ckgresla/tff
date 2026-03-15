"""Evaluation utilities for enwik8 content-type analysis.

Tags enwik8 byte sequences by content type (XML, prose, numbers, etc.)
and computes per-type BPC. For the ChoosyTransformer, also analyzes
routing patterns per content type.
"""

import logging
from collections import defaultdict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int

from choosy.data import Enwik8Dataset

log = logging.getLogger("choosy.eval")

# Content type tags
TAG_XML = 0
TAG_PROSE = 1
TAG_NUMBER = 2
TAG_WHITESPACE = 3
TAG_ENTITY = 4

CONTENT_TYPE_NAMES = {
    TAG_XML: "xml",
    TAG_PROSE: "prose",
    TAG_NUMBER: "number",
    TAG_WHITESPACE: "whitespace",
    TAG_ENTITY: "entity",
}


def tag_enwik8_bytes(data: np.ndarray) -> np.ndarray:
    """Tag each byte position with a content type.

    Uses a simple state machine to identify:
      0 = xml_tag     (inside <...>)
      1 = prose       (regular text, default)
      2 = number      (digit sequences)
      3 = whitespace  (spaces, tabs, newlines)
      4 = entity      (inside &...;)

    Args:
        data: raw byte array (uint8)

    Returns:
        tags: array of same length, dtype uint8
    """
    tags = np.full(len(data), TAG_PROSE, dtype=np.uint8)
    in_tag = False
    in_entity = False

    for i in range(len(data)):
        b = data[i]

        # XML tag detection
        if b == ord('<'):
            in_tag = True
        if in_tag:
            tags[i] = TAG_XML
            if b == ord('>'):
                in_tag = False
            continue

        # Entity detection (&amp; &lt; &#NNN;)
        if b == ord('&'):
            in_entity = True
        if in_entity:
            tags[i] = TAG_ENTITY
            if b == ord(';'):
                in_entity = False
            continue

        # Simple classifications
        if ord('0') <= b <= ord('9'):
            tags[i] = TAG_NUMBER
        elif b in (ord(' '), ord('\t'), ord('\n'), ord('\r')):
            tags[i] = TAG_WHITESPACE
        # else: stays TAG_PROSE (default)

    return tags


def _per_token_loss(input_seq, target_seq, *, model):
    """Compute per-token cross-entropy loss (not averaged)."""
    result = model(input_seq, dropout_key=None)
    if isinstance(result, tuple):
        logits, _ = result
    else:
        logits = result
    return optax.softmax_cross_entropy_with_integer_labels(logits, target_seq)


def _per_token_loss_with_routing(input_seq, target_seq, *, model):
    """Compute per-token loss AND return router logits for routing analysis."""
    result = model(input_seq, dropout_key=None)
    if isinstance(result, tuple):
        logits, router_logits = result
    else:
        logits = result
        router_logits = jnp.zeros((1, 1))  # dummy for non-routing models
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_seq)
    return loss, router_logits


def evaluate_by_content_type(
    model,
    data_path: str,
    seq_len: int = 256,
    max_sequences: int = 2000,
) -> dict:
    """Evaluate BPC stratified by content type on enwik8 test set.

    Processes sequential (non-overlapping) chunks of the test set,
    tags each byte position, and aggregates per-token loss by tag.

    Args:
        model: trained model (any type)
        data_path: path to enwik8.zip
        seq_len: sequence length for evaluation
        max_sequences: max number of sequences to evaluate

    Returns:
        dict with:
          "overall_bpc": float
          "per_type": {type_name: {"bpc": float, "count": int}}
          "tag_distribution": {type_name: fraction}
    """
    # Load test data and tag it
    test_dataset = Enwik8Dataset(data_path, seq_len=seq_len, split="test")
    test_data = test_dataset.data
    tags = tag_enwik8_bytes(test_data)
    log.info("Tagged %d test bytes", len(tags))

    # Log tag distribution
    for tag_id, tag_name in CONTENT_TYPE_NAMES.items():
        count = int(np.sum(tags == tag_id))
        log.info("  %s: %d (%.1f%%)", tag_name, count, 100 * count / len(tags))

    # JIT the per-token loss function
    loss_fn = partial(_per_token_loss, model=model)
    batched_loss_fn = eqx.filter_jit(jax.vmap(loss_fn))

    # Sequential evaluation over non-overlapping chunks
    losses_by_type = defaultdict(list)
    all_losses = []

    num_sequences = min(max_sequences, (len(test_data) - 1) // seq_len)
    batch_size = 32
    num_batches = (num_sequences + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_seq = batch_idx * batch_size
        end_seq = min(start_seq + batch_size, num_sequences)
        actual_batch = end_seq - start_seq

        # Build batch of sequential chunks
        inputs_list = []
        targets_list = []
        for seq_idx in range(start_seq, end_seq):
            offset = seq_idx * seq_len
            inputs_list.append(test_data[offset:offset + seq_len])
            targets_list.append(test_data[offset + 1:offset + seq_len + 1])

        inputs = jnp.array(np.stack(inputs_list))
        targets = jnp.array(np.stack(targets_list))

        # Per-token losses: (batch, seq_len)
        per_token = batched_loss_fn(inputs, targets)

        # Aggregate by content type
        for seq_idx in range(actual_batch):
            offset = (start_seq + seq_idx) * seq_len
            # Tags for TARGET positions (shifted by 1 from input)
            seq_tags = tags[offset + 1:offset + seq_len + 1]
            seq_losses = np.array(per_token[seq_idx])

            all_losses.append(seq_losses)

            for tag_id, tag_name in CONTENT_TYPE_NAMES.items():
                mask = seq_tags == tag_id
                if mask.any():
                    losses_by_type[tag_name].append(seq_losses[mask])

    # Compute BPC per type
    overall_loss = np.mean(np.concatenate(all_losses))
    overall_bpc = float(overall_loss / np.log(2))

    per_type = {}
    for tag_name in CONTENT_TYPE_NAMES.values():
        if tag_name in losses_by_type:
            type_losses = np.concatenate(losses_by_type[tag_name])
            per_type[tag_name] = {
                "bpc": float(np.mean(type_losses) / np.log(2)),
                "count": len(type_losses),
            }

    # Tag distribution
    tag_dist = {}
    for tag_id, tag_name in CONTENT_TYPE_NAMES.items():
        tag_dist[tag_name] = float(np.sum(tags == tag_id)) / len(tags)

    results = {
        "overall_bpc": overall_bpc,
        "per_type": per_type,
        "tag_distribution": tag_dist,
    }

    # Print results
    log.info("Overall BPC: %.4f", overall_bpc)
    for tag_name, info in sorted(per_type.items(), key=lambda x: x[1]["bpc"]):
        log.info("  %-12s BPC: %.4f  (%d tokens)", tag_name, info["bpc"], info["count"])

    return results


def evaluate_routing_patterns(
    model,
    data_path: str,
    seq_len: int = 256,
    max_sequences: int = 500,
) -> dict:
    """Analyze routing patterns by content type for ChoosyTransformer.

    Returns a heatmap: layer_usage[content_type, layer_id] showing
    which layers fire for which content types.

    Args:
        model: ChoosyTransformer model
        data_path: path to enwik8.zip
        seq_len: sequence length
        max_sequences: max sequences to analyze

    Returns:
        dict with:
          "layer_usage": {type_name: [usage_fraction_per_layer]}
          "routing_entropy": float (avg entropy of routing decisions)
          "overall_layer_usage": [usage_fraction_per_layer]
    """
    test_dataset = Enwik8Dataset(data_path, seq_len=seq_len, split="test")
    test_data = test_dataset.data
    tags = tag_enwik8_bytes(test_data)

    # JIT the routing-aware loss function
    loss_fn = partial(_per_token_loss_with_routing, model=model)
    batched_fn = eqx.filter_jit(jax.vmap(loss_fn))

    # We get router_logits of shape (K, N) per sequence
    # For routing analysis, we look at argmax routing decisions
    num_sequences = min(max_sequences, (len(test_data) - 1) // seq_len)
    batch_size = 32

    # Collect routing decisions per content type
    # route_counts[tag_name] = array of shape (N,) counting layer selections
    num_pool = model.num_pool_layers
    num_steps = model.num_routing_steps
    route_counts = {name: np.zeros(num_pool) for name in CONTENT_TYPE_NAMES.values()}
    total_counts = np.zeros(num_pool)
    all_entropies = []

    num_batches = (num_sequences + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_seq = batch_idx * batch_size
        end_seq = min(start_seq + batch_size, num_sequences)
        actual_batch = end_seq - start_seq

        inputs_list = []
        targets_list = []
        for seq_idx in range(start_seq, end_seq):
            offset = seq_idx * seq_len
            inputs_list.append(test_data[offset:offset + seq_len])
            targets_list.append(test_data[offset + 1:offset + seq_len + 1])

        inputs = jnp.array(np.stack(inputs_list))
        targets = jnp.array(np.stack(targets_list))

        _, router_logits_batch = batched_fn(inputs, targets)
        # router_logits_batch: (batch, K, N)
        router_logits_np = np.array(router_logits_batch)

        for seq_idx in range(actual_batch):
            offset = (start_seq + seq_idx) * seq_len
            seq_tags = tags[offset:offset + seq_len]

            # Determine dominant content type for this sequence
            tag_counts = np.bincount(seq_tags, minlength=5)
            dominant_tag = int(np.argmax(tag_counts))
            dominant_name = CONTENT_TYPE_NAMES[dominant_tag]

            # Routing decisions: argmax over each step's logits
            seq_router = router_logits_np[seq_idx]  # (K, N)
            choices = np.argmax(seq_router, axis=-1)  # (K,)

            for choice in choices:
                route_counts[dominant_name][choice] += 1
                total_counts[choice] += 1

            # Routing entropy
            probs = np.exp(seq_router) / np.exp(seq_router).sum(axis=-1, keepdims=True)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
            all_entropies.append(np.mean(entropy))

    # Normalize to fractions
    layer_usage = {}
    for tag_name, counts in route_counts.items():
        total = counts.sum()
        if total > 0:
            layer_usage[tag_name] = (counts / total).tolist()
        else:
            layer_usage[tag_name] = [0.0] * num_pool

    overall_usage = (total_counts / total_counts.sum()).tolist() if total_counts.sum() > 0 else [0.0] * num_pool
    avg_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0

    results = {
        "layer_usage": layer_usage,
        "routing_entropy": avg_entropy,
        "overall_layer_usage": overall_usage,
    }

    # Print
    log.info("Routing entropy: %.4f (max = %.4f)", avg_entropy, np.log(num_pool))
    log.info("Overall layer usage: %s", [f"{u:.2f}" for u in overall_usage])
    for tag_name, usage in sorted(layer_usage.items()):
        log.info("  %-12s %s", tag_name, [f"{u:.2f}" for u in usage])

    return results

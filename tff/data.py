"""Data loading for byte-level language modeling."""

import zipfile
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray
from typing import Iterator


class Enwik8Dataset:
    """Byte-level enwik8 dataset loader."""

    def __init__(self, data_path: str, seq_len: int = 256, split: str = "train"):
        """
        Load enwik8 dataset.

        Args:
            data_path: Path to enwik8.zip
            seq_len: Sequence length for training
            split: One of 'train', 'valid', 'test'
        """
        self.seq_len = seq_len

        # Load data from zip - read as raw bytes
        with zipfile.ZipFile(data_path, 'r') as zf:
            data_bytes = np.frombuffer(zf.read('enwik8'), dtype=np.uint8)

        # Standard enwik8 splits
        # Train: first 90M bytes
        # Valid: next 5M bytes
        # Test: last 5M bytes
        total_len = len(data_bytes)
        train_end = 90_000_000
        valid_end = 95_000_000

        if split == "train":
            self.data = data_bytes[:train_end]
        elif split == "valid":
            self.data = data_bytes[train_end:valid_end]
        elif split == "test":
            self.data = data_bytes[valid_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Loaded {split} split: {len(self.data):,} bytes")

    def get_batch(self, batch_size: int, key: PRNGKeyArray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a random batch of sequences.

        Args:
            batch_size: Number of sequences in batch
            key: JAX PRNG key for sampling

        Returns:
            inputs: [batch_size, seq_len] of byte values
            targets: [batch_size, seq_len] of byte values (shifted by 1)
        """
        # Randomly sample starting positions using JAX
        max_start = len(self.data) - self.seq_len - 1
        starts = jr.randint(key, shape=(batch_size,), minval=0, maxval=max_start)

        # Convert to numpy for indexing
        starts_np = np.array(starts)

        # Extract sequences
        inputs = np.stack([
            self.data[start:start + self.seq_len]
            for start in starts_np
        ])

        targets = np.stack([
            self.data[start + 1:start + self.seq_len + 1]
            for start in starts_np
        ])

        # Convert to JAX arrays
        return jnp.array(inputs), jnp.array(targets)

    def iterate_batches(
        self,
        batch_size: int,
        key: PRNGKeyArray,
        num_batches: int = None
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Iterate over batches.

        Args:
            batch_size: Batch size
            key: JAX PRNG key
            num_batches: Number of batches (None = infinite)

        Yields:
            (inputs, targets) tuples
        """
        count = 0
        while True:
            if num_batches is not None and count >= num_batches:
                break

            # Split key for each batch
            key, batch_key = jr.split(key)
            yield self.get_batch(batch_size, batch_key)
            count += 1

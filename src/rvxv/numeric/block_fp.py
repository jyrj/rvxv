from __future__ import annotations

import math

import numpy as np


class BlockFloatingPoint:
    """Block Floating Point: shared exponent across a group of elements.

    Each block has N elements sharing a single exponent. The exponent is
    chosen as the maximum exponent across all elements in the block.
    Elements are stored as signed mantissas with the shared exponent.
    """

    def __init__(self, element_bits: int = 8, group_size: int = 32):
        self.element_bits = element_bits
        self.group_size = group_size
        self.mantissa_bits = element_bits - 1  # 1 sign bit
        self.max_mantissa = (1 << self.mantissa_bits) - 1

    def encode_block(self, values: np.ndarray) -> tuple[int, np.ndarray]:
        """Encode a block of float values.

        Returns:
            (shared_exponent, mantissa_array) where mantissa_array contains
            signed integer mantissas scaled by the shared exponent.
        """
        if len(values) > self.group_size:
            raise ValueError(f"Block size {len(values)} exceeds group size {self.group_size}")

        # Find shared exponent (max absolute value)
        abs_max = np.max(np.abs(values[values != 0])) if np.any(values != 0) else 0.0

        if abs_max == 0:
            return 0, np.zeros(len(values), dtype=np.int8)

        shared_exp = math.floor(math.log2(abs_max))
        scale = 2.0 ** (shared_exp - self.mantissa_bits + 1)

        # Quantize each element
        mantissas = np.clip(
            np.round(values / scale).astype(np.int32),
            -self.max_mantissa, self.max_mantissa
        ).astype(np.int8)

        return shared_exp, mantissas

    def decode_block(self, shared_exp: int, mantissas: np.ndarray) -> np.ndarray:
        """Decode a block from shared exponent + mantissas to float values."""
        scale = 2.0 ** (shared_exp - self.mantissa_bits + 1)
        return mantissas.astype(np.float64) * scale

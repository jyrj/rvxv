"""Tests for BFloat16 numeric type."""

from __future__ import annotations

import math
import struct

import pytest

from rvxv.numeric.bfloat16 import BFloat16
from rvxv.numeric.rounding import RoundingMode


@pytest.fixture
def bf16():
    return BFloat16()


class TestBFloat16:
    def test_from_fp32_exact(self):
        """Values exactly representable in BF16 should roundtrip perfectly."""
        bits = BFloat16.from_fp32(1.0)
        assert bits == 0x3F80
        assert BFloat16.to_fp32(bits) == 1.0

    def test_from_fp32_zero(self):
        assert BFloat16.from_fp32(0.0) == 0x0000
        assert BFloat16.from_fp32(-0.0) == 0x8000

    def test_from_fp32_infinity(self):
        assert BFloat16.from_fp32(float("inf")) == 0x7F80
        assert BFloat16.from_fp32(float("-inf")) == 0xFF80

    def test_from_fp32_nan(self):
        bits = BFloat16.from_fp32(float("nan"))
        assert bits == 0x7FC0

    def test_to_fp32_exact(self):
        """BF16 to FP32 is always exact (just shift left by 16)."""
        assert BFloat16.to_fp32(0x3F80) == 1.0
        assert BFloat16.to_fp32(0x4000) == 2.0
        assert BFloat16.to_fp32(0x0000) == 0.0

    def test_rounding_rne_tie_to_even(self):
        """RNE: when exactly halfway, round to even LSB."""
        # Construct FP32 value where lower 16 bits = 0x8000 (exact halfway)
        # with LSB=0 (even) → should NOT round up
        fp32_bits = (0x3F80 << 16) | 0x8000  # 1.0 + exact halfway
        value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
        result = BFloat16.from_fp32(value, RoundingMode.RNE)
        assert result == 0x3F80  # Even LSB, tie → keep

    def test_rounding_rne_tie_to_even_up(self):
        """RNE: when exactly halfway with odd LSB, round up."""
        fp32_bits = (0x3F81 << 16) | 0x8000  # LSB=1 (odd) + exact halfway
        value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
        result = BFloat16.from_fp32(value, RoundingMode.RNE)
        assert result == 0x3F82  # Odd LSB, tie → round up

    def test_rounding_rtz(self):
        """RTZ: always truncate."""
        fp32_bits = (0x3F80 << 16) | 0xFFFF  # Just below next BF16 value
        value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
        result = BFloat16.from_fp32(value, RoundingMode.RTZ)
        assert result == 0x3F80  # Truncated

    def test_roundtrip_standard_values(self, bf16):
        """Values that are exact in BF16 should roundtrip."""
        for val in [0.0, 1.0, -1.0, 2.0, 0.5, 256.0, -0.125]:
            bits = BFloat16.from_fp32(val)
            roundtrip = BFloat16.to_fp32(bits)
            assert roundtrip == val, f"Roundtrip failed for {val}"

    def test_decode(self, bf16):
        assert bf16.decode(0x3F80) == 1.0
        assert bf16.decode(0x4000) == 2.0
        assert bf16.decode(0x0000) == 0.0
        assert math.isinf(bf16.decode(0x7F80))
        assert math.isnan(bf16.decode(0x7FC0))

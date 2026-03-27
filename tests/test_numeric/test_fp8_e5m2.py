"""Tests for FP8 E5M2 numeric type — IEEE 754-like with infinity and NaN."""

from __future__ import annotations

import math

import pytest

from rvxv.numeric.fp8_e5m2 import FP8_E5M2_FORMAT, FP8E5M2


@pytest.fixture
def fp8():
    return FP8E5M2()


class TestFP8E5M2:
    def test_format_constants(self):
        assert FP8_E5M2_FORMAT.total_bits == 8
        assert FP8_E5M2_FORMAT.exponent_bits == 5
        assert FP8_E5M2_FORMAT.mantissa_bits == 2
        assert FP8_E5M2_FORMAT.bias == 15
        assert FP8_E5M2_FORMAT.has_infinity is True
        assert FP8_E5M2_FORMAT.has_nan is True
        assert FP8_E5M2_FORMAT.max_finite == 57344.0

    def test_decode_zero(self, fp8):
        assert fp8.decode(0x00) == 0.0

    def test_decode_one(self, fp8):
        # 1.0 = (1+0/4) × 2^(15-15) = 1.0, exp=15 => 0x3C
        assert fp8.decode(0x3C) == 1.0

    def test_decode_positive_infinity(self, fp8):
        # exp=31, mant=0 → +Inf
        assert math.isinf(fp8.decode(0x7C))
        assert fp8.decode(0x7C) > 0

    def test_decode_negative_infinity(self, fp8):
        assert math.isinf(fp8.decode(0xFC))
        assert fp8.decode(0xFC) < 0

    def test_decode_nan(self, fp8):
        # exp=31, mant!=0
        assert math.isnan(fp8.decode(0x7D))
        assert math.isnan(fp8.decode(0x7E))
        assert math.isnan(fp8.decode(0x7F))

    def test_decode_max_normal(self, fp8):
        # 0x7B = exp=30, mant=3: (1 + 3/4) × 2^(30-15) = 1.75 × 32768 = 57344.0
        assert fp8.decode(0x7B) == 57344.0

    def test_decode_min_subnormal(self, fp8):
        # 0x01 = exp=0, mant=1: (1/4) × 2^(1-15) = 0.25 × 2^-14 = 2^-16
        assert fp8.decode(0x01) == pytest.approx(2**-16)

    @pytest.mark.numeric
    def test_exhaustive_inf_count(self, fp8):
        """Exactly 2 infinity encodings: 0x7C (+Inf) and 0xFC (-Inf)."""
        inf_patterns = [b for b in range(256) if math.isinf(fp8.decode(b))]
        assert inf_patterns == [0x7C, 0xFC]

    @pytest.mark.numeric
    def test_exhaustive_nan_count(self, fp8):
        """6 NaN encodings: 0x7D,0x7E,0x7F and 0xFD,0xFE,0xFF."""
        nan_patterns = [b for b in range(256) if math.isnan(fp8.decode(b))]
        assert nan_patterns == [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]

    def test_encode_infinity(self, fp8):
        bits = fp8.encode(float("inf"))
        assert bits == 0x7C

    def test_encode_nan(self, fp8):
        bits = fp8.encode(float("nan"))
        assert math.isnan(fp8.decode(bits))

"""Tests for FP8 E4M3 numeric type — the most critical format.

FP8 E4M3 has NO infinity (saturates to ±448), only 2 NaN encodings (0x7F, 0xFF),
and supports subnormals. These tests exhaustively verify every bit pattern.
"""

from __future__ import annotations

import math

import pytest

from rvxv.numeric.fp8_e4m3 import FP8_E4M3_FORMAT, FP8E4M3


@pytest.fixture
def fp8():
    return FP8E4M3()


class TestFP8E4M3Format:
    def test_format_constants(self):
        assert FP8_E4M3_FORMAT.total_bits == 8
        assert FP8_E4M3_FORMAT.exponent_bits == 4
        assert FP8_E4M3_FORMAT.mantissa_bits == 3
        assert FP8_E4M3_FORMAT.bias == 7
        assert FP8_E4M3_FORMAT.has_infinity is False
        assert FP8_E4M3_FORMAT.has_nan is True
        assert FP8_E4M3_FORMAT.has_subnormals is True
        assert FP8_E4M3_FORMAT.max_finite == 448.0


class TestFP8E4M3Decode:
    def test_zero(self, fp8):
        assert fp8.decode(0x00) == 0.0
        assert math.copysign(1, fp8.decode(0x00)) > 0  # positive zero

    def test_negative_zero(self, fp8):
        val = fp8.decode(0x80)
        assert val == 0.0
        assert math.copysign(1, val) < 0  # negative zero

    def test_one(self, fp8):
        # 1.0 = (-1)^0 × (1 + 0/8) × 2^(7-7) = 1.0
        assert fp8.decode(0x38) == 1.0

    def test_max_positive(self, fp8):
        # 0x7E = exp=15, mant=6: (1 + 6/8) × 2^(15-7) = 1.75 × 256 = 448.0
        assert fp8.decode(0x7E) == 448.0

    def test_max_negative(self, fp8):
        assert fp8.decode(0xFE) == -448.0

    def test_nan_positive(self, fp8):
        assert math.isnan(fp8.decode(0x7F))

    def test_nan_negative(self, fp8):
        assert math.isnan(fp8.decode(0xFF))

    def test_min_subnormal(self, fp8):
        # 0x01 = exp=0, mant=1: (1/8) × 2^(1-7) = 0.125 × 2^-6 = 2^-9
        val = fp8.decode(0x01)
        assert val == pytest.approx(2**-9)

    def test_max_subnormal(self, fp8):
        # 0x07 = exp=0, mant=7: (7/8) × 2^(1-7) = 0.875 × 2^-6
        val = fp8.decode(0x07)
        assert val == pytest.approx(7 * 2**-9)

    def test_min_normal(self, fp8):
        # 0x08 = exp=1, mant=0: (1 + 0/8) × 2^(1-7) = 2^-6
        assert fp8.decode(0x08) == pytest.approx(2**-6)

    def test_two(self, fp8):
        # 2.0 = 1.0 × 2^1, so exp=8 (8-7=1), mant=0: 0x40
        assert fp8.decode(0x40) == 2.0

    @pytest.mark.numeric
    def test_exhaustive_no_nan_except_7f(self, fp8):
        """Verify only 0x7F and 0xFF are NaN."""
        nan_patterns = []
        for bits in range(256):
            if math.isnan(fp8.decode(bits)):
                nan_patterns.append(bits)
        assert nan_patterns == [0x7F, 0xFF]

    @pytest.mark.numeric
    def test_exhaustive_no_infinity(self, fp8):
        """E4M3 has no infinity — all decoded values must be finite or NaN."""
        for bits in range(256):
            val = fp8.decode(bits)
            assert not math.isinf(val), f"Unexpected infinity at bits=0x{bits:02X}"

    @pytest.mark.numeric
    def test_exhaustive_sign_bit(self, fp8):
        """Bit 7 controls sign for all non-NaN values."""
        for bits in range(128):
            pos = fp8.decode(bits)
            neg = fp8.decode(bits | 0x80)
            if math.isnan(pos):
                continue
            if pos == 0.0:
                # Both zeros should differ only in sign
                assert neg == 0.0
            else:
                assert neg == pytest.approx(-pos)


class TestFP8E4M3Encode:
    def test_encode_zero(self, fp8):
        assert fp8.encode(0.0) == 0x00

    def test_encode_neg_zero(self, fp8):
        assert fp8.encode(-0.0) == 0x80

    def test_encode_one(self, fp8):
        assert fp8.encode(1.0) == 0x38

    def test_encode_max(self, fp8):
        assert fp8.encode(448.0) == 0x7E

    def test_encode_nan(self, fp8):
        assert fp8.encode(float("nan")) == 0x7F

    def test_encode_saturate_overflow(self, fp8):
        """Values > 448 should saturate to 0x7E (448.0), NOT produce infinity."""
        result = fp8.encode(500.0)
        assert result == 0x7E, f"Expected saturation to 0x7E, got 0x{result:02X}"

    def test_encode_saturate_infinity(self, fp8):
        """Infinity should saturate to max finite."""
        result = fp8.encode(float("inf"))
        assert result == 0x7E

    def test_encode_neg_saturate(self, fp8):
        result = fp8.encode(-500.0)
        assert result == 0xFE  # -448.0

    def test_encode_no_nan_encoding(self, fp8):
        """Encoding should never accidentally produce the NaN bit pattern 0x7F."""
        for val in [447.0, 448.0, 449.0, 500.0, 1000.0]:
            bits = fp8.encode(val)
            assert (bits & 0x7F) != 0x7F, f"Accidental NaN for value {val}"

    @pytest.mark.numeric
    def test_roundtrip_all_values(self, fp8):
        """decode(encode(decode(bits))) == decode(bits) for all 256 bit patterns."""
        for bits in range(256):
            original = fp8.decode(bits)
            if math.isnan(original):
                continue
            re_encoded = fp8.encode(original)
            re_decoded = fp8.decode(re_encoded)
            msg = (
                f"Roundtrip failed: bits=0x{bits:02X}, val={original},"
                f" re_encoded=0x{re_encoded:02X}, re_decoded={re_decoded}"
            )
            assert re_decoded == pytest.approx(original, abs=1e-10), msg

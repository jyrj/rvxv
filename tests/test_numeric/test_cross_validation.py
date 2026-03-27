"""Cross-validate RVXV numeric library against independent references.

These tests verify RVXV's numeric types produce correct results by comparing
against:
  - struct.pack/unpack (Python's IEEE 754 implementation)
  - numpy float16/float32 bit patterns
  - Hardcoded known IEEE 754 bit patterns from the specification
  - Manual computation from the format definition

NO RVXV code is used to compute expected values — every expected value is
derived from an independent source.
"""

import math
import struct

import numpy as np
import pytest

from rvxv.numeric.bfloat16 import BFloat16
from rvxv.numeric.fp8_e4m3 import FP8E4M3
from rvxv.numeric.fp8_e5m2 import FP8E5M2
from rvxv.numeric.int4 import Int4Signed, Int4Unsigned

# -----------------------------------------------------------------------
# BFloat16 cross-validation against struct.pack (IEEE 754 FP32 truncation)
# -----------------------------------------------------------------------


class TestBFloat16CrossValidation:
    """Cross-validate BFloat16 against Python struct (IEEE 754)."""

    @staticmethod
    def _ieee754_bf16(value: float) -> int:
        """Compute BF16 encoding independently using struct.pack.

        BFloat16 is simply the upper 16 bits of an IEEE 754 float32,
        with round-to-nearest-even applied to the truncated bits.
        """
        fp32_bits = struct.unpack(">I", struct.pack(">f", value))[0]
        upper16 = (fp32_bits >> 16) & 0xFFFF
        # Round-to-nearest-even: check bits [15:0] of fp32
        lower16 = fp32_bits & 0xFFFF
        if lower16 > 0x8000:
            upper16 += 1
        elif lower16 == 0x8000:
            # Tie: round to even (bit 0 of upper16)
            if upper16 & 1:
                upper16 += 1
        return upper16 & 0xFFFF

    @pytest.mark.parametrize("value,expected_bits", [
        (1.0, 0x3F80),      # IEEE 754: 0x3F800000 -> upper 16 = 0x3F80
        (-1.0, 0xBF80),     # IEEE 754: 0xBF800000
        (0.0, 0x0000),
        (2.0, 0x4000),      # IEEE 754: 0x40000000
        (0.5, 0x3F00),      # IEEE 754: 0x3F000000
        (float("inf"), 0x7F80),
        (float("-inf"), 0xFF80),
    ])
    def test_known_ieee754_values(self, value, expected_bits):
        """BFloat16 encoding matches known IEEE 754 bit patterns."""
        # Verify our reference is correct
        fp32_bits = struct.unpack(">I", struct.pack(">f", value))[0]
        assert (fp32_bits >> 16) == expected_bits, (
            f"Reference check failed: FP32(0x{fp32_bits:08X}) >> 16 != 0x{expected_bits:04X}"
        )
        # Now verify RVXV
        result = BFloat16.from_fp32(value)
        assert result == expected_bits, (
            f"BFloat16.from_fp32({value}) = 0x{result:04X}, "
            f"expected 0x{expected_bits:04X} (from IEEE 754)"
        )

    def test_roundtrip_matches_numpy(self):
        """BFloat16 -> FP32 roundtrip matches numpy's float32 bit interpretation."""
        test_values = [1.0, -1.0, 0.5, 2.0, 3.14, 100.0, 1e-5, 1e30]
        for val in test_values:
            bf16_bits = BFloat16.from_fp32(val)
            recovered = BFloat16.to_fp32(bf16_bits)
            # The recovered value should be the same as truncating FP32 to upper 16 bits
            bf16_ref = self._ieee754_bf16(val)
            # Reconstruct FP32 from BF16 reference
            ref_fp32_bits = bf16_ref << 16
            ref_val = struct.unpack(">f", struct.pack(">I", ref_fp32_bits))[0]
            assert recovered == pytest.approx(ref_val, rel=1e-6), (
                f"BFloat16 roundtrip({val}): got {recovered}, "
                f"independent ref gives {ref_val}"
            )

    def test_rounding_tie_to_even(self):
        """BFloat16 correctly rounds ties to even (RNE).

        3.14159 in FP32: 0x40490FDB
        Upper 16: 0x4049, lower 16: 0x0FDB
        0x0FDB < 0x8000, so no rounding up -> result is 0x4049 = 3.140625

        Construct a value that hits the tie case (lower16 == 0x8000).
        """
        # 0x3FC08000 has lower16 == 0x8000, upper16 = 0x3FC0 (even) -> no round
        tie_even_bits = 0x3FC08000
        tie_even_val = struct.unpack(">f", struct.pack(">I", tie_even_bits))[0]
        result = BFloat16.from_fp32(tie_even_val)
        assert result == 0x3FC0, f"Tie-to-even: expected 0x3FC0, got 0x{result:04X}"

        # 0x3FC18000 has lower16 == 0x8000, upper16 = 0x3FC1 (odd) -> round up
        tie_odd_bits = 0x3FC18000
        tie_odd_val = struct.unpack(">f", struct.pack(">I", tie_odd_bits))[0]
        result = BFloat16.from_fp32(tie_odd_val)
        assert result == 0x3FC2, f"Tie-to-even (odd): expected 0x3FC2, got 0x{result:04X}"

    def test_nan_preserved(self):
        """NaN values are preserved through BFloat16 encoding."""
        result = BFloat16.from_fp32(float("nan"))
        recovered = BFloat16.to_fp32(result)
        assert math.isnan(recovered), "NaN not preserved through BFloat16 roundtrip"


# -----------------------------------------------------------------------
# FP8 E4M3 cross-validation against manual IEEE-like computation
# -----------------------------------------------------------------------


class TestFP8E4M3CrossValidation:
    """Cross-validate FP8 E4M3 against manual format definition.

    FP8 E4M3 format (per OCP spec / NVIDIA H100):
      - 1 sign bit, 4 exponent bits, 3 mantissa bits
      - Bias = 7
      - No infinity: max exponent encodes finite values, not inf
      - NaN: only exp=15, mant=7 (0x7F and 0xFF)
      - Max finite: (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448.0
    """

    @staticmethod
    def _manual_decode(bits: int) -> float:
        """Decode FP8 E4M3 from scratch, no RVXV code."""
        sign = (bits >> 7) & 1
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7

        if exp == 15 and mant == 7:
            return float("nan")
        if exp == 0 and mant == 0:
            return -0.0 if sign else 0.0
        if exp == 0:
            # Subnormal: (mant/8) * 2^(1-7) = mant * 2^(-9)
            val = mant * (2 ** -9)
        else:
            # Normal: (1 + mant/8) * 2^(exp-7)
            val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        return -val if sign else val

    def test_exhaustive_decode_matches_manual(self):
        """All 256 FP8 E4M3 bit patterns match manual decode."""
        fp8 = FP8E4M3()
        mismatches = []
        for bits in range(256):
            rvxv_val = fp8.decode(bits)
            manual_val = self._manual_decode(bits)

            if math.isnan(rvxv_val) and math.isnan(manual_val):
                continue
            if rvxv_val == 0.0 and manual_val == 0.0:
                # Check sign of zero
                if math.copysign(1, rvxv_val) != math.copysign(1, manual_val):
                    mismatches.append(f"0x{bits:02X}: sign mismatch on zero")
                continue
            if rvxv_val != pytest.approx(manual_val, abs=1e-12):
                mismatches.append(
                    f"0x{bits:02X}: RVXV={rvxv_val}, manual={manual_val}"
                )
        assert not mismatches, "Decode mismatches:\n" + "\n".join(mismatches)

    @pytest.mark.parametrize("value,expected_bits", [
        (0.0, 0x00),
        (1.0, 0x38),      # exp=7, mant=0: (1+0/8)*2^0 = 1.0
        (2.0, 0x40),      # exp=8, mant=0: (1+0/8)*2^1 = 2.0
        (0.5, 0x30),      # exp=6, mant=0: (1+0/8)*2^-1 = 0.5
        (448.0, 0x7E),    # max finite
        (-448.0, 0xFE),   # max negative
        (1.5, 0x3C),      # exp=7, mant=4: (1+4/8)*2^0 = 1.5
        (0.25, 0x28),     # exp=5, mant=0: (1+0/8)*2^-2 = 0.25
    ])
    def test_known_encodings(self, value, expected_bits):
        """Spot-check encodings against manually computed bit patterns."""
        fp8 = FP8E4M3()
        result = fp8.encode(value)
        # Verify the expected bits decode to the right value
        assert self._manual_decode(expected_bits) == pytest.approx(value, abs=1e-10), (
            f"Reference check: 0x{expected_bits:02X} should decode to {value}"
        )
        assert result == expected_bits, (
            f"FP8E4M3.encode({value}) = 0x{result:02X}, expected 0x{expected_bits:02X}"
        )


# -----------------------------------------------------------------------
# FP8 E5M2 cross-validation
# -----------------------------------------------------------------------


class TestFP8E5M2CrossValidation:
    """Cross-validate FP8 E5M2 against IEEE 754 half-precision structure.

    FP8 E5M2 is structurally similar to IEEE 754 (has infinity and NaN):
      - 1 sign, 5 exponent, 2 mantissa bits
      - Bias = 15
      - Infinity: exp=31, mant=0
      - NaN: exp=31, mant!=0
      - Max finite: (1 + 2/4) * 2^(30-15) = 1.5 * 32768 = 49152... wait
        Actually max finite: (1 + 3/4) * 2^(30-15) = 1.75 * 32768 = 57344
    """

    @staticmethod
    def _manual_decode(bits: int) -> float:
        """Decode FP8 E5M2 from scratch."""
        sign = (bits >> 7) & 1
        exp = (bits >> 2) & 0x1F
        mant = bits & 0x3

        if exp == 31:
            if mant == 0:
                return float("-inf") if sign else float("inf")
            return float("nan")
        if exp == 0 and mant == 0:
            return -0.0 if sign else 0.0
        if exp == 0:
            val = (mant / 4.0) * (2.0 ** (1 - 15))
        else:
            val = (1.0 + mant / 4.0) * (2.0 ** (exp - 15))
        return -val if sign else val

    def test_exhaustive_decode_matches_manual(self):
        """All 256 FP8 E5M2 bit patterns match manual decode."""
        fp8 = FP8E5M2()
        mismatches = []
        for bits in range(256):
            rvxv_val = fp8.decode(bits)
            manual_val = self._manual_decode(bits)

            if math.isnan(rvxv_val) and math.isnan(manual_val):
                continue
            if math.isinf(rvxv_val) and math.isinf(manual_val):
                assert math.copysign(1, rvxv_val) == math.copysign(1, manual_val)
                continue
            if rvxv_val == 0.0 and manual_val == 0.0:
                if math.copysign(1, rvxv_val) != math.copysign(1, manual_val):
                    mismatches.append(f"0x{bits:02X}: sign mismatch on zero")
                continue
            if rvxv_val != pytest.approx(manual_val, abs=1e-12):
                mismatches.append(
                    f"0x{bits:02X}: RVXV={rvxv_val}, manual={manual_val}"
                )
        assert not mismatches, "Decode mismatches:\n" + "\n".join(mismatches)

    def test_max_finite_is_57344(self):
        """Max finite E5M2 value: (1 + 3/4) * 2^15 = 57344."""
        fp8 = FP8E5M2()
        val = fp8.decode(0x7B)  # exp=30, mant=3
        manual = (1.0 + 3 / 4.0) * (2.0 ** 15)
        assert manual == 57344.0
        assert val == 57344.0

    def test_infinity_exists(self):
        """E5M2 has infinity (unlike E4M3)."""
        fp8 = FP8E5M2()
        assert math.isinf(fp8.decode(0x7C))   # +inf
        assert math.isinf(fp8.decode(0xFC))   # -inf


# -----------------------------------------------------------------------
# INT4 cross-validation
# -----------------------------------------------------------------------


class TestInt4CrossValidation:
    """Cross-validate INT4 against two's complement definition."""

    def test_signed_range(self):
        """Signed INT4: 4-bit two's complement range is [-8, 7]."""
        for val in range(-8, 8):
            bits = Int4Signed.encode(val)
            decoded = Int4Signed.decode(bits)
            assert decoded == val, f"Roundtrip failed for {val}"
            # Verify against two's complement definition
            if val >= 0:
                assert bits == val
            else:
                assert bits == (val + 16) & 0xF  # two's complement

    def test_unsigned_range(self):
        """Unsigned INT4: range is [0, 15]."""
        for val in range(16):
            bits = Int4Unsigned.encode(val)
            decoded = Int4Unsigned.decode(bits)
            assert decoded == val
            assert bits == val  # trivial identity for unsigned


# -----------------------------------------------------------------------
# Cross-format consistency: BFloat16 vs numpy float32
# -----------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Verify numeric types are consistent with numpy's IEEE 754."""

    def test_bf16_to_fp32_matches_numpy(self):
        """BFloat16 -> FP32 reconstruction matches numpy bit interpretation."""
        test_bits = [0x3F80, 0x4000, 0x3F00, 0xBF80, 0x7F80, 0x0000]
        for bf16_bits in test_bits:
            rvxv_val = BFloat16.to_fp32(bf16_bits)
            # Independent: reconstruct from bits
            fp32_bits = bf16_bits << 16
            np_val = np.array([fp32_bits], dtype=np.uint32).view(np.float32)[0]
            if math.isnan(rvxv_val) and math.isnan(np_val):
                continue
            assert rvxv_val == pytest.approx(float(np_val), abs=1e-10), (
                f"BF16 0x{bf16_bits:04X}: RVXV={rvxv_val}, numpy={np_val}"
            )

    def test_fp8e4m3_values_in_fp32_range(self):
        """Every FP8 E4M3 decoded value is exactly representable in FP32."""
        fp8 = FP8E4M3()
        for bits in range(256):
            val = fp8.decode(bits)
            if math.isnan(val):
                continue
            # FP8 E4M3 values have at most 4 significant bits (1 implicit + 3 mantissa)
            # so they must be exactly representable in FP32 (24-bit mantissa)
            fp32_bits = struct.unpack(">I", struct.pack(">f", val))[0]
            recovered = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
            assert recovered == val, (
                f"FP8 0x{bits:02X} value {val} not exactly representable in FP32"
            )

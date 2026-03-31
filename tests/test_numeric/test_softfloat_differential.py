"""Exhaustive differential testing of RVXV numeric library against Berkeley SoftFloat.

This is the core research contribution: proving RVXV's AI numeric types are
bit-accurate by comparing every representable value against the IEEE 754
gold-standard reference implementation used by the RISC-V ecosystem.

Tests cover:
  - FP8 E4M3: all 256 bit patterns (decode + encode roundtrip)
  - FP8 E5M2: all 256 bit patterns (decode + encode roundtrip)
  - BFloat16: all 65536 bit patterns (decode) + 10000 random FP32 encode tests
  - Rounding modes: RNE, RTZ, RTP, RTN for BFloat16 encode

Total: 66,048 exhaustive comparisons + 50,000 rounding mode tests.
"""

from __future__ import annotations

import math
import struct
import random

import pytest

from rvxv.numeric.bfloat16 import BFloat16
from rvxv.numeric.fp8_e4m3 import FP8E4M3
from rvxv.numeric.fp8_e5m2 import FP8E5M2
from rvxv.numeric.rounding import RoundingMode

try:
    from rvxv.numeric.softfloat_wrapper import SoftFloat
    _sf = SoftFloat()
    HAS_SOFTFLOAT = True
except OSError:
    HAS_SOFTFLOAT = False

skipif_no_softfloat = pytest.mark.skipif(
    not HAS_SOFTFLOAT,
    reason="Berkeley SoftFloat library not built (run: cd extern/spike/build && make)",
)


# -----------------------------------------------------------------------
# FP8 E4M3: Exhaustive differential testing (256 patterns)
# -----------------------------------------------------------------------


@skipif_no_softfloat
class TestFP8E4M3vsSoftFloat:
    """Compare every FP8 E4M3 value against Berkeley SoftFloat."""

    def test_exhaustive_decode_256_patterns(self):
        """Decode all 256 FP8 E4M3 bit patterns and compare against SoftFloat.

        SoftFloat path: e4m3 -> bf16 -> f32 (two conversions).
        RVXV path: e4m3 -> float (direct decode).
        """
        fp8 = FP8E4M3()
        mismatches = []

        for bits in range(256):
            rvxv_val = fp8.decode(bits)
            sf_val = _sf.e4m3_to_f32(bits)

            if math.isnan(rvxv_val) and math.isnan(sf_val):
                continue
            if rvxv_val == 0.0 and sf_val == 0.0:
                if math.copysign(1, rvxv_val) != math.copysign(1, sf_val):
                    mismatches.append(
                        f"0x{bits:02X}: sign mismatch on zero "
                        f"(RVXV={'-' if math.copysign(1,rvxv_val)<0 else '+'}0, "
                        f"SF={'-' if math.copysign(1,sf_val)<0 else '+'}0)"
                    )
                continue
            if rvxv_val != pytest.approx(sf_val, abs=1e-12):
                mismatches.append(
                    f"0x{bits:02X}: RVXV={rvxv_val}, SoftFloat={sf_val}"
                )

        assert not mismatches, (
            f"FP8 E4M3 decode mismatches vs SoftFloat ({len(mismatches)}/256):\n"
            + "\n".join(mismatches[:20])
        )

    def test_exhaustive_encode_roundtrip(self):
        """For all 256 patterns: decode with SoftFloat, encode with RVXV,
        verify we get the same bits back."""
        fp8 = FP8E4M3()
        _sf.set_rounding_mode(SoftFloat.ROUND_NEAR_EVEN)
        mismatches = []

        for bits in range(256):
            sf_val = _sf.e4m3_to_f32(bits)
            if math.isnan(sf_val):
                continue

            rvxv_bits = fp8.encode(sf_val)
            if rvxv_bits != bits:
                mismatches.append(
                    f"0x{bits:02X}: decode={sf_val}, re-encode=0x{rvxv_bits:02X}"
                )

        assert not mismatches, (
            f"FP8 E4M3 roundtrip mismatches ({len(mismatches)}):\n"
            + "\n".join(mismatches[:20])
        )

    def test_f32_to_e4m3_matches_softfloat(self):
        """Encode FP32 -> E4M3 via both RVXV and SoftFloat, compare."""
        fp8 = FP8E4M3()
        _sf.set_rounding_mode(SoftFloat.ROUND_NEAR_EVEN)

        test_values = [
            0.0, 1.0, -1.0, 0.5, 2.0, 448.0, -448.0,
            0.25, 1.5, 3.0, 100.0, 0.001953125,  # min subnormal
            500.0, -500.0, 1000.0,  # overflow -> saturate
        ]
        mismatches = []

        for val in test_values:
            rvxv_bits = fp8.encode(val)
            sf_bits = _sf.f32_to_e4m3(val, saturate=True)
            if rvxv_bits != sf_bits:
                mismatches.append(
                    f"{val}: RVXV=0x{rvxv_bits:02X}, SoftFloat=0x{sf_bits:02X}"
                )

        assert not mismatches, (
            f"f32->E4M3 mismatches:\n" + "\n".join(mismatches)
        )


# -----------------------------------------------------------------------
# FP8 E5M2: Exhaustive differential testing (256 patterns)
# -----------------------------------------------------------------------


@skipif_no_softfloat
class TestFP8E5M2vsSoftFloat:
    """Compare every FP8 E5M2 value against Berkeley SoftFloat."""

    def test_exhaustive_decode_256_patterns(self):
        """Decode all 256 FP8 E5M2 bit patterns and compare against SoftFloat."""
        fp8 = FP8E5M2()
        mismatches = []

        for bits in range(256):
            rvxv_val = fp8.decode(bits)
            sf_val = _sf.e5m2_to_f32(bits)

            if math.isnan(rvxv_val) and math.isnan(sf_val):
                continue
            if math.isinf(rvxv_val) and math.isinf(sf_val):
                if math.copysign(1, rvxv_val) != math.copysign(1, sf_val):
                    mismatches.append(f"0x{bits:02X}: inf sign mismatch")
                continue
            if rvxv_val == 0.0 and sf_val == 0.0:
                if math.copysign(1, rvxv_val) != math.copysign(1, sf_val):
                    mismatches.append(f"0x{bits:02X}: zero sign mismatch")
                continue
            if rvxv_val != pytest.approx(sf_val, abs=1e-12):
                mismatches.append(
                    f"0x{bits:02X}: RVXV={rvxv_val}, SoftFloat={sf_val}"
                )

        assert not mismatches, (
            f"FP8 E5M2 decode mismatches vs SoftFloat ({len(mismatches)}/256):\n"
            + "\n".join(mismatches[:20])
        )

    def test_exhaustive_encode_roundtrip(self):
        """Decode with SoftFloat, encode with RVXV, verify same bits."""
        fp8 = FP8E5M2()
        _sf.set_rounding_mode(SoftFloat.ROUND_NEAR_EVEN)
        mismatches = []

        for bits in range(256):
            sf_val = _sf.e5m2_to_f32(bits)
            if math.isnan(sf_val) or math.isinf(sf_val):
                continue
            rvxv_bits = fp8.encode(sf_val)
            if rvxv_bits != bits:
                mismatches.append(
                    f"0x{bits:02X}: decode={sf_val}, re-encode=0x{rvxv_bits:02X}"
                )

        assert not mismatches, (
            f"FP8 E5M2 roundtrip mismatches ({len(mismatches)}):\n"
            + "\n".join(mismatches[:20])
        )


# -----------------------------------------------------------------------
# BFloat16: Exhaustive differential testing (65536 patterns)
# -----------------------------------------------------------------------


@skipif_no_softfloat
class TestBFloat16vsSoftFloat:
    """Compare all 65536 BFloat16 values against Berkeley SoftFloat."""

    def test_exhaustive_decode_65536_patterns(self):
        """Decode all 65536 BFloat16 bit patterns via both RVXV and SoftFloat."""
        mismatches = []

        for bits in range(65536):
            rvxv_val = BFloat16.to_fp32(bits)
            sf_val = _sf.bf16_to_f32(bits)

            if math.isnan(rvxv_val) and math.isnan(sf_val):
                continue
            if math.isinf(rvxv_val) and math.isinf(sf_val):
                if math.copysign(1, rvxv_val) != math.copysign(1, sf_val):
                    mismatches.append(f"0x{bits:04X}: inf sign mismatch")
                continue
            if rvxv_val == 0.0 and sf_val == 0.0:
                if math.copysign(1, rvxv_val) != math.copysign(1, sf_val):
                    mismatches.append(f"0x{bits:04X}: zero sign mismatch")
                continue
            if rvxv_val != pytest.approx(sf_val, abs=1e-45):
                mismatches.append(
                    f"0x{bits:04X}: RVXV={rvxv_val}, SoftFloat={sf_val}"
                )

        assert not mismatches, (
            f"BFloat16 decode mismatches vs SoftFloat ({len(mismatches)}/65536):\n"
            + "\n".join(mismatches[:20])
        )

    def test_f32_to_bf16_rne_10000_random(self):
        """Encode 10000 random FP32 values to BFloat16, compare RVXV vs SoftFloat (RNE)."""
        _sf.set_rounding_mode(SoftFloat.ROUND_NEAR_EVEN)
        rng = random.Random(42)
        mismatches = []

        for _ in range(10000):
            # Generate random FP32 bit pattern
            f32_bits = rng.randint(0, 0xFFFFFFFF)
            f32_val = struct.unpack("f", struct.pack("I", f32_bits))[0]

            if math.isnan(f32_val) or math.isinf(f32_val):
                continue

            rvxv_bits = BFloat16.from_fp32(f32_val, RoundingMode.RNE)
            sf_bits = _sf.f32_bits_to_bf16(f32_bits)

            if rvxv_bits != sf_bits:
                mismatches.append(
                    f"FP32=0x{f32_bits:08X} ({f32_val}): "
                    f"RVXV=0x{rvxv_bits:04X}, SoftFloat=0x{sf_bits:04X}"
                )

        assert not mismatches, (
            f"f32->BF16 (RNE) mismatches ({len(mismatches)}/10000):\n"
            + "\n".join(mismatches[:20])
        )

    @pytest.mark.parametrize("sf_mode,rvxv_mode", [
        (SoftFloat.ROUND_NEAR_EVEN, RoundingMode.RNE),
        (SoftFloat.ROUND_MINMAG, RoundingMode.RTZ),
        (SoftFloat.ROUND_MAX, RoundingMode.RTP),
        (SoftFloat.ROUND_MIN, RoundingMode.RTN),
        (SoftFloat.ROUND_NEAR_MAXMAG, RoundingMode.RNA),
    ])
    def test_f32_to_bf16_all_rounding_modes(self, sf_mode, rvxv_mode):
        """Test FP32->BF16 encoding across all 5 IEEE rounding modes."""
        _sf.set_rounding_mode(sf_mode)
        rng = random.Random(123)
        mismatches = []

        for _ in range(10000):
            f32_bits = rng.randint(0, 0xFFFFFFFF)
            f32_val = struct.unpack("f", struct.pack("I", f32_bits))[0]

            if math.isnan(f32_val) or math.isinf(f32_val):
                continue

            rvxv_bits = BFloat16.from_fp32(f32_val, rvxv_mode)
            sf_bits = _sf.f32_bits_to_bf16(f32_bits)

            if rvxv_bits != sf_bits:
                mismatches.append(
                    f"FP32=0x{f32_bits:08X}: RVXV=0x{rvxv_bits:04X}, SF=0x{sf_bits:04X}"
                )

        assert not mismatches, (
            f"f32->BF16 ({rvxv_mode.value}) mismatches ({len(mismatches)}/10000):\n"
            + "\n".join(mismatches[:20])
        )


# -----------------------------------------------------------------------
# Summary statistics (for paper)
# -----------------------------------------------------------------------


@skipif_no_softfloat
class TestDifferentialSummary:
    """Produce a summary of all differential testing for the paper."""

    def test_print_summary(self):
        """Print summary of differential testing coverage."""
        fp8_e4m3 = FP8E4M3()
        fp8_e5m2 = FP8E5M2()

        e4m3_match = 0
        for bits in range(256):
            rvxv = fp8_e4m3.decode(bits)
            sf = _sf.e4m3_to_f32(bits)
            if math.isnan(rvxv) and math.isnan(sf):
                e4m3_match += 1
            elif rvxv == sf:
                e4m3_match += 1
            elif rvxv == 0.0 and sf == 0.0:
                e4m3_match += 1

        e5m2_match = 0
        for bits in range(256):
            rvxv = fp8_e5m2.decode(bits)
            sf = _sf.e5m2_to_f32(bits)
            if math.isnan(rvxv) and math.isnan(sf):
                e5m2_match += 1
            elif math.isinf(rvxv) and math.isinf(sf):
                e5m2_match += 1
            elif rvxv == sf:
                e5m2_match += 1
            elif rvxv == 0.0 and sf == 0.0:
                e5m2_match += 1

        bf16_match = 0
        for bits in range(65536):
            rvxv = BFloat16.to_fp32(bits)
            sf = _sf.bf16_to_f32(bits)
            if math.isnan(rvxv) and math.isnan(sf):
                bf16_match += 1
            elif math.isinf(rvxv) and math.isinf(sf):
                bf16_match += 1
            elif rvxv == sf:
                bf16_match += 1
            elif rvxv == 0.0 and sf == 0.0:
                bf16_match += 1

        print(f"\n{'='*60}")
        print("DIFFERENTIAL TESTING SUMMARY (RVXV vs Berkeley SoftFloat)")
        print(f"{'='*60}")
        print(f"FP8 E4M3 decode: {e4m3_match}/256 patterns match ({e4m3_match/256*100:.1f}%)")
        print(f"FP8 E5M2 decode: {e5m2_match}/256 patterns match ({e5m2_match/256*100:.1f}%)")
        print(f"BFloat16 decode: {bf16_match}/65536 patterns match ({bf16_match/65536*100:.1f}%)")
        print(f"Total exhaustive comparisons: {256+256+65536}")
        print(f"{'='*60}")

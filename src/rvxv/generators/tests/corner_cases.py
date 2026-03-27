"""AI-specific corner case database for RISC-V verification.

Test vectors that exercise numeric edge cases in every
AI-relevant data type.  Each corner case is a named test vector with operand
bit patterns and (optionally) expected result bit patterns.

The test vectors returned by :func:`get_corner_cases` are *raw bit-pattern*
integers.  Floating-point values are represented by their in-memory encoding
(e.g. ``0x7E`` for the FP8-E4M3 maximum of 448.0).  Integer values are
represented as unsigned bit patterns that match the element width (e.g. the
INT8 value -128 is stored as ``0x80``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rvxv.core.type_system import ElementType

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class CornerCase:
    """A single corner case test vector."""

    name: str
    description: str
    operand_a: list[int]  # Raw bit patterns for first source
    operand_b: list[int]  # Raw bit patterns for second source
    expected: list[int] = field(default_factory=list)  # Expected result (may be empty)


# ---------------------------------------------------------------------------
# Helpers for generating repeated / padded vectors
# ---------------------------------------------------------------------------


def _repeat(val: int, n: int = 16) -> list[int]:
    """Return a list of *n* copies of *val*."""
    return [val] * n


def _tile(vals: list[int], n: int = 16) -> list[int]:
    """Tile *vals* to length *n*."""
    return (vals * ((n // len(vals)) + 1))[:n]


def _signed_to_unsigned(val: int, bits: int) -> int:
    """Convert a signed Python int to its unsigned N-bit representation."""
    if val < 0:
        return val + (1 << bits)
    return val & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# INT8 corner cases
# ---------------------------------------------------------------------------


def _int8_dot_product_cases() -> list[CornerCase]:
    """Corner cases for INT8 dot-product operations.

    A dot-product of k INT8 elements produces partial products in the range
    [-128*127, 127*127] = [-16256, 16129] per pair, accumulated into INT32.
    """
    s = lambda v: _signed_to_unsigned(v, 8)  # noqa: E731

    return [
        CornerCase(
            name="int8_zero_times_max",
            description="zero x anything = 0",
            operand_a=_repeat(s(0)),
            operand_b=_repeat(s(127)),
            expected=_repeat(0, 4),  # 32-bit result elements
        ),
        CornerCase(
            name="int8_max_times_max",
            description="127 x 127 = 16129 (fits in INT32)",
            operand_a=_repeat(s(127)),
            operand_b=_repeat(s(127)),
        ),
        CornerCase(
            name="int8_min_times_min",
            description="-128 x -128 = 16384 (positive, fits in INT32)",
            operand_a=_repeat(s(-128)),
            operand_b=_repeat(s(-128)),
        ),
        CornerCase(
            name="int8_mixed_sign",
            description="-128 x 127 = -16256",
            operand_a=_repeat(s(-128)),
            operand_b=_repeat(s(127)),
        ),
        CornerCase(
            name="int8_alternating_signs",
            description="Alternating +127 / -128 to test accumulator sign changes",
            operand_a=_tile([s(127), s(-128)]),
            operand_b=_tile([s(1), s(1)]),
        ),
        CornerCase(
            name="int8_all_zeros",
            description="Both operands zero -- result must be exactly 0",
            operand_a=_repeat(s(0)),
            operand_b=_repeat(s(0)),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="int8_accum_overflow_positive",
            description=(
                "Large positive partial products that stress INT32 accumulator "
                "upper range: 4 x (127*127)=64516"
            ),
            operand_a=_repeat(s(127)),
            operand_b=_repeat(s(127)),
        ),
        CornerCase(
            name="int8_accum_overflow_negative",
            description=(
                "Large negative accumulation: 4 x (-128*127) = -65024, "
                "verify no unsigned wrap"
            ),
            operand_a=_repeat(s(-128)),
            operand_b=_repeat(s(127)),
        ),
        CornerCase(
            name="int8_one_times_value",
            description="Identity: 1 x N = N",
            operand_a=_repeat(s(1)),
            operand_b=[s(v) for v in range(-128, -128 + 16)],
        ),
        CornerCase(
            name="int8_minus_one_times_value",
            description="Negation: -1 x N = -N",
            operand_a=_repeat(s(-1)),
            operand_b=[s(v) for v in range(0, 16)],
        ),
    ]


# ---------------------------------------------------------------------------
# UINT8 corner cases
# ---------------------------------------------------------------------------


def _uint8_dot_product_cases() -> list[CornerCase]:
    """Corner cases for UINT8 dot-product operations."""
    return [
        CornerCase(
            name="uint8_zero_times_max",
            description="0 x 255 = 0",
            operand_a=_repeat(0),
            operand_b=_repeat(255),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="uint8_max_times_max",
            description="255 x 255 = 65025",
            operand_a=_repeat(255),
            operand_b=_repeat(255),
        ),
        CornerCase(
            name="uint8_all_ones",
            description="1 x 1 = 1",
            operand_a=_repeat(1),
            operand_b=_repeat(1),
        ),
        CornerCase(
            name="uint8_ramp",
            description="Ramp values 0..15",
            operand_a=list(range(16)),
            operand_b=_repeat(1),
        ),
    ]


# ---------------------------------------------------------------------------
# INT4 corner cases
# ---------------------------------------------------------------------------


def _int4_cases() -> list[CornerCase]:
    """Corner cases for signed INT4 operations.

    INT4 values are packed two per byte.  The valid range is [-8, +7].
    Bit patterns use the lower nibble (bits 3:0) for the value.
    """
    s4 = lambda v: _signed_to_unsigned(v, 4)  # noqa: E731

    return [
        CornerCase(
            name="int4_boundary_min",
            description="INT4 minimum: -8 (0x8)",
            operand_a=_repeat(s4(-8)),
            operand_b=_repeat(s4(1)),
        ),
        CornerCase(
            name="int4_boundary_max",
            description="INT4 maximum: +7 (0x7)",
            operand_a=_repeat(s4(7)),
            operand_b=_repeat(s4(7)),
        ),
        CornerCase(
            name="int4_zero",
            description="Zero operand",
            operand_a=_repeat(s4(0)),
            operand_b=_repeat(s4(7)),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="int4_min_times_min",
            description="-8 x -8 = 64",
            operand_a=_repeat(s4(-8)),
            operand_b=_repeat(s4(-8)),
        ),
        CornerCase(
            name="int4_mixed_sign",
            description="-8 x 7 = -56",
            operand_a=_repeat(s4(-8)),
            operand_b=_repeat(s4(7)),
        ),
        CornerCase(
            name="int4_packed_aliasing",
            description=(
                "Adjacent nibbles with opposite signs to test packed "
                "element extraction: [-8, 7, -8, 7, ...]"
            ),
            operand_a=_tile([s4(-8), s4(7)]),
            operand_b=_tile([s4(1), s4(1)]),
        ),
    ]


# ---------------------------------------------------------------------------
# UINT4 corner cases
# ---------------------------------------------------------------------------


def _uint4_cases() -> list[CornerCase]:
    """Corner cases for unsigned UINT4 operations (range 0..15)."""
    return [
        CornerCase(
            name="uint4_boundary_zero",
            description="UINT4 zero",
            operand_a=_repeat(0),
            operand_b=_repeat(15),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="uint4_boundary_max",
            description="UINT4 maximum: 15 x 15 = 225",
            operand_a=_repeat(15),
            operand_b=_repeat(15),
        ),
        CornerCase(
            name="uint4_ramp",
            description="Ramp values 0..15",
            operand_a=list(range(16)),
            operand_b=_repeat(1),
        ),
    ]


# ---------------------------------------------------------------------------
# FP8 E4M3 corner cases
# ---------------------------------------------------------------------------


def _fp8_e4m3_cases() -> list[CornerCase]:
    """Corner cases for FP8 E4M3 format.

    E4M3 key properties:
      - NO infinity representation (overflow saturates to +/-448)
      - Only two NaN encodings: 0x7F (+NaN), 0xFF (-NaN)
      - Max positive: 0x7E = 448.0
      - Min subnormal: 0x01 = 2^-9
      - Zero: 0x00, Negative zero: 0x80
    """
    return [
        CornerCase(
            name="fp8e4m3_max_times_max",
            description="448.0 x 448.0 = 200704 (max positive * max positive)",
            operand_a=_repeat(0x7E),
            operand_b=_repeat(0x7E),
        ),
        CornerCase(
            name="fp8e4m3_min_subnormal",
            description="Min subnormal (2^-9 = ~0.00195) x 1.0",
            operand_a=_repeat(0x01),
            operand_b=_repeat(0x38),  # 1.0 in E4M3
        ),
        CornerCase(
            name="fp8e4m3_subnormal_times_subnormal",
            description="Subnormal x subnormal -- result is very small, may flush to zero",
            operand_a=_repeat(0x01),  # min subnormal
            operand_b=_repeat(0x01),  # min subnormal
        ),
        CornerCase(
            name="fp8e4m3_zero",
            description="Zero x max -- must produce exactly zero",
            operand_a=_repeat(0x00),
            operand_b=_repeat(0x7E),
        ),
        CornerCase(
            name="fp8e4m3_neg_zero",
            description="Negative zero (0x80) -- sign handling",
            operand_a=_repeat(0x80),
            operand_b=_repeat(0x7E),
        ),
        CornerCase(
            name="fp8e4m3_nan",
            description="NaN (0x7F) operand -- NaN must propagate",
            operand_a=_repeat(0x7F),
            operand_b=_repeat(0x38),  # 1.0
        ),
        CornerCase(
            name="fp8e4m3_neg_nan",
            description="Negative NaN (0xFF) operand",
            operand_a=_repeat(0xFF),
            operand_b=_repeat(0x38),
        ),
        CornerCase(
            name="fp8e4m3_saturation",
            description=(
                "Result exceeds 448.0 -- should clamp to max finite, "
                "NOT produce infinity (E4M3 has no Inf)"
            ),
            operand_a=_repeat(0x7E),  # 448.0
            operand_b=_repeat(0x40),  # 2.0
        ),
        CornerCase(
            name="fp8e4m3_opposite_signs",
            description="Positive x negative -- result must be negative",
            operand_a=_repeat(0x7E),  # +448.0
            operand_b=_repeat(0xFE),  # -448.0
        ),
        CornerCase(
            name="fp8e4m3_one",
            description="1.0 x 1.0 = 1.0 (identity)",
            operand_a=_repeat(0x38),  # 1.0
            operand_b=_repeat(0x38),  # 1.0
        ),
        CornerCase(
            name="fp8e4m3_all_subnormals",
            description="All subnormal values (0x01..0x07)",
            operand_a=_tile([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x01]),
            operand_b=_repeat(0x38),  # 1.0
        ),
    ]


# ---------------------------------------------------------------------------
# FP8 E5M2 corner cases
# ---------------------------------------------------------------------------


def _fp8_e5m2_cases() -> list[CornerCase]:
    """Corner cases for FP8 E5M2 format.

    E5M2 key properties:
      - IEEE 754 compatible: has infinity AND NaN
      - +Inf: 0x7C, -Inf: 0xFC
      - NaN: 0x7D, 0x7E, 0x7F (and their sign-flipped counterparts)
      - Max normal: 0x7B = 57344.0
      - Zero: 0x00, Negative zero: 0x80
      - Min subnormal: 0x01 = 2^-16
    """
    return [
        CornerCase(
            name="fp8e5m2_pos_inf",
            description="Positive infinity (0x7C) operand",
            operand_a=_repeat(0x7C),
            operand_b=_repeat(0x3C),  # 1.0 in E5M2
        ),
        CornerCase(
            name="fp8e5m2_neg_inf",
            description="Negative infinity (0xFC) operand",
            operand_a=_repeat(0xFC),
            operand_b=_repeat(0x3C),  # 1.0
        ),
        CornerCase(
            name="fp8e5m2_nan_0x7d",
            description="NaN encoding 0x7D -- must propagate",
            operand_a=_repeat(0x7D),
            operand_b=_repeat(0x3C),
        ),
        CornerCase(
            name="fp8e5m2_nan_0x7e",
            description="NaN encoding 0x7E -- must propagate",
            operand_a=_repeat(0x7E),
            operand_b=_repeat(0x3C),
        ),
        CornerCase(
            name="fp8e5m2_nan_0x7f",
            description="NaN encoding 0x7F -- must propagate",
            operand_a=_repeat(0x7F),
            operand_b=_repeat(0x3C),
        ),
        CornerCase(
            name="fp8e5m2_max_normal",
            description="Max normal (0x7B = 57344.0) x max normal",
            operand_a=_repeat(0x7B),
            operand_b=_repeat(0x7B),
        ),
        CornerCase(
            name="fp8e5m2_zero_times_inf",
            description="0 x Inf = NaN (invalid operation)",
            operand_a=_repeat(0x00),
            operand_b=_repeat(0x7C),  # +Inf
        ),
        CornerCase(
            name="fp8e5m2_nan_propagation",
            description="NaN x finite -- NaN must propagate to result",
            operand_a=_repeat(0x7F),  # NaN
            operand_b=_repeat(0x7B),  # max normal
        ),
        CornerCase(
            name="fp8e5m2_inf_minus_inf",
            description="Inf + (-Inf) = NaN (for reduction operations)",
            operand_a=_tile([0x7C, 0xFC]),  # +Inf, -Inf alternating
            operand_b=_repeat(0x3C),  # 1.0
        ),
        CornerCase(
            name="fp8e5m2_subnormal",
            description="Min subnormal (0x01 = 2^-16) operations",
            operand_a=_repeat(0x01),
            operand_b=_repeat(0x3C),  # 1.0
        ),
        CornerCase(
            name="fp8e5m2_neg_zero",
            description="Negative zero (0x80) handling",
            operand_a=_repeat(0x80),
            operand_b=_repeat(0x3C),
        ),
        CornerCase(
            name="fp8e5m2_opposite_inf",
            description="+Inf x -1 = -Inf",
            operand_a=_repeat(0x7C),  # +Inf
            operand_b=_repeat(0xBC),  # -1.0
        ),
    ]


# ---------------------------------------------------------------------------
# BF16 corner cases
# ---------------------------------------------------------------------------


def _bf16_cases() -> list[CornerCase]:
    """Corner cases for BFloat16 format.

    BF16 is a truncation of FP32 to 16 bits with 8 exponent and 7 mantissa bits.
    Same exponent range as FP32.

    Key bit patterns:
      - +0:      0x0000
      - -0:      0x8000
      - +1.0:    0x3F80
      - -1.0:    0xBF80
      - +Inf:    0x7F80
      - -Inf:    0xFF80
      - NaN:     0x7FC0 (canonical), any 0x7F81..0x7FFF
      - Max:     0x7F7F = 3.3895e+38
      - Min normal:     0x0080 = 2^-126
      - Min subnormal:  0x0001 = 2^-133
    """
    return [
        # --- Rounding mode boundaries ---
        CornerCase(
            name="bf16_round_exact_halfway",
            description=(
                "Exact halfway point: FP32 value whose lower 16 bits are "
                "0x8000 (guard=1, round=0, sticky=0) -- exercises RNE tie-to-even"
            ),
            operand_a=_repeat(0x3F80, 8),  # 1.0
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_round_just_above_halfway",
            description=(
                "Just above halfway: tests that round-up occurs for "
                "guard=1, round=0, sticky=1"
            ),
            operand_a=_repeat(0x3F81, 8),  # 1.0 + 1 ULP in BF16
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_round_just_below_halfway",
            description=(
                "Just below halfway: tests that truncation occurs for "
                "guard=0, round=1, sticky=1"
            ),
            operand_a=_repeat(0x3F7F, 8),  # 1.0 - 1 ULP
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_round_tie_to_even_up",
            description="Tie-to-even case where LSB=1 (should round up)",
            operand_a=_repeat(0x3F81, 8),
            operand_b=_repeat(0x3F81, 8),
        ),
        CornerCase(
            name="bf16_round_tie_to_even_down",
            description="Tie-to-even case where LSB=0 (should round down / truncate)",
            operand_a=_repeat(0x3F80, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        # --- Special values ---
        CornerCase(
            name="bf16_subnormal",
            description="Min subnormal (0x0001 = 2^-133)",
            operand_a=_repeat(0x0001, 8),
            operand_b=_repeat(0x3F80, 8),  # 1.0
        ),
        CornerCase(
            name="bf16_min_normal",
            description="Min normal (0x0080 = 2^-126)",
            operand_a=_repeat(0x0080, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_max_finite",
            description="Max finite (0x7F7F ~= 3.39e+38)",
            operand_a=_repeat(0x7F7F, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_pos_inf",
            description="Positive infinity (0x7F80)",
            operand_a=_repeat(0x7F80, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_neg_inf",
            description="Negative infinity (0xFF80)",
            operand_a=_repeat(0xFF80, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_nan",
            description="NaN (0x7FC0 canonical) -- must propagate",
            operand_a=_repeat(0x7FC0, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_inf_times_zero",
            description="Inf x 0 = NaN (invalid operation)",
            operand_a=_repeat(0x7F80, 8),  # +Inf
            operand_b=_repeat(0x0000, 8),  # 0
        ),
        CornerCase(
            name="bf16_neg_zero",
            description="Negative zero (0x8000)",
            operand_a=_repeat(0x8000, 8),
            operand_b=_repeat(0x3F80, 8),
        ),
        CornerCase(
            name="bf16_precision_loss_boundary",
            description=(
                "FP32 -> BF16 precision loss boundary: value that is exactly "
                "representable in FP32 but not in BF16"
            ),
            operand_a=_repeat(0x3F80, 8),  # 1.0 exactly
            operand_b=_repeat(0x3380, 8),  # small value where precision matters
        ),
    ]


# ---------------------------------------------------------------------------
# INT16 corner cases
# ---------------------------------------------------------------------------


def _int16_cases() -> list[CornerCase]:
    """Corner cases for signed INT16 operations (range -32768 to +32767)."""
    s = lambda v: _signed_to_unsigned(v, 16)  # noqa: E731

    return [
        CornerCase(
            name="int16_zero_times_max",
            description="zero x anything = 0",
            operand_a=_repeat(s(0), 8),
            operand_b=_repeat(s(32767), 8),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="int16_max_times_max",
            description="32767 x 32767 = 1073676289",
            operand_a=_repeat(s(32767), 8),
            operand_b=_repeat(s(32767), 8),
        ),
        CornerCase(
            name="int16_min_times_min",
            description="-32768 x -32768 = 1073741824 (positive)",
            operand_a=_repeat(s(-32768), 8),
            operand_b=_repeat(s(-32768), 8),
        ),
        CornerCase(
            name="int16_mixed_sign",
            description="-32768 x 32767 = -1073709056",
            operand_a=_repeat(s(-32768), 8),
            operand_b=_repeat(s(32767), 8),
        ),
        CornerCase(
            name="int16_alternating_signs",
            description="Alternating +32767 / -32768 for accumulator sign changes",
            operand_a=_tile([s(32767), s(-32768)], 8),
            operand_b=_tile([s(1), s(1)], 8),
        ),
        CornerCase(
            name="int16_identity",
            description="1 x N = N (identity multiplication)",
            operand_a=_repeat(s(1), 8),
            operand_b=[s(v) for v in [-32768, -1, 0, 1, 127, 32767, -128, 256]],
        ),
    ]


# ---------------------------------------------------------------------------
# UINT16 corner cases
# ---------------------------------------------------------------------------


def _uint16_cases() -> list[CornerCase]:
    """Corner cases for unsigned UINT16 operations (range 0..65535)."""
    return [
        CornerCase(
            name="uint16_zero_times_max",
            description="0 x 65535 = 0",
            operand_a=_repeat(0, 8),
            operand_b=_repeat(65535, 8),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="uint16_max_times_max",
            description="65535 x 65535 = 4294836225",
            operand_a=_repeat(65535, 8),
            operand_b=_repeat(65535, 8),
        ),
        CornerCase(
            name="uint16_one_times_value",
            description="1 x N = N",
            operand_a=_repeat(1, 8),
            operand_b=[0, 1, 255, 256, 32767, 32768, 65534, 65535],
        ),
        CornerCase(
            name="uint16_ramp",
            description="Ramp values 0..7",
            operand_a=list(range(8)),
            operand_b=_repeat(1, 8),
        ),
    ]


# ---------------------------------------------------------------------------
# INT32 corner cases
# ---------------------------------------------------------------------------


def _int32_cases() -> list[CornerCase]:
    """Corner cases for signed INT32 operations."""
    s = lambda v: _signed_to_unsigned(v, 32)  # noqa: E731

    return [
        CornerCase(
            name="int32_zero",
            description="0 x max = 0",
            operand_a=_repeat(s(0), 4),
            operand_b=_repeat(s(2147483647), 4),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="int32_max",
            description="INT32_MAX x 1",
            operand_a=_repeat(s(2147483647), 4),
            operand_b=_repeat(s(1), 4),
        ),
        CornerCase(
            name="int32_min",
            description="INT32_MIN x 1",
            operand_a=_repeat(s(-2147483648), 4),
            operand_b=_repeat(s(1), 4),
        ),
        CornerCase(
            name="int32_mixed_sign",
            description="INT32_MIN x -1 (potential overflow to INT64)",
            operand_a=_repeat(s(-2147483648), 4),
            operand_b=_repeat(s(-1), 4),
        ),
        CornerCase(
            name="int32_negation",
            description="-1 x N = -N",
            operand_a=_repeat(s(-1), 4),
            operand_b=[s(v) for v in [0, 1, -1, 2147483647]],
        ),
    ]


# ---------------------------------------------------------------------------
# UINT32 corner cases
# ---------------------------------------------------------------------------


def _uint32_cases() -> list[CornerCase]:
    """Corner cases for unsigned UINT32 operations."""
    return [
        CornerCase(
            name="uint32_zero",
            description="0 x max = 0",
            operand_a=_repeat(0, 4),
            operand_b=_repeat(0xFFFFFFFF, 4),
            expected=_repeat(0, 4),
        ),
        CornerCase(
            name="uint32_max",
            description="UINT32_MAX x 1",
            operand_a=_repeat(0xFFFFFFFF, 4),
            operand_b=_repeat(1, 4),
        ),
        CornerCase(
            name="uint32_max_squared",
            description="UINT32_MAX x UINT32_MAX (large product)",
            operand_a=_repeat(0xFFFFFFFF, 4),
            operand_b=_repeat(0xFFFFFFFF, 4),
        ),
        CornerCase(
            name="uint32_identity",
            description="1 x N = N",
            operand_a=_repeat(1, 4),
            operand_b=[0, 1, 0x7FFFFFFF, 0xFFFFFFFF],
        ),
    ]


# ---------------------------------------------------------------------------
# FP32 corner cases
# ---------------------------------------------------------------------------


def _fp32_cases() -> list[CornerCase]:
    """Corner cases for IEEE FP32 format."""
    return [
        CornerCase(
            name="fp32_max_finite",
            description="Max finite (3.4028e+38) x 1.0",
            operand_a=_repeat(0x7F7FFFFF, 4),
            operand_b=_repeat(0x3F800000, 4),  # 1.0
        ),
        CornerCase(
            name="fp32_min_normal",
            description="Min normal (1.1755e-38) x 1.0",
            operand_a=_repeat(0x00800000, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
        CornerCase(
            name="fp32_subnormal",
            description="Min subnormal (2^-149) x 1.0",
            operand_a=_repeat(0x00000001, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
        CornerCase(
            name="fp32_pos_inf",
            description="+Inf x 1.0 = +Inf",
            operand_a=_repeat(0x7F800000, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
        CornerCase(
            name="fp32_neg_inf",
            description="-Inf x 1.0 = -Inf",
            operand_a=_repeat(0xFF800000, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
        CornerCase(
            name="fp32_nan",
            description="NaN propagation (quiet NaN x 1.0)",
            operand_a=_repeat(0x7FC00000, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
        CornerCase(
            name="fp32_zero_times_inf",
            description="0.0 x Inf = NaN (invalid)",
            operand_a=_repeat(0x00000000, 4),
            operand_b=_repeat(0x7F800000, 4),
        ),
        CornerCase(
            name="fp32_neg_zero",
            description="Negative zero x 1.0",
            operand_a=_repeat(0x80000000, 4),
            operand_b=_repeat(0x3F800000, 4),
        ),
    ]


# ---------------------------------------------------------------------------
# FP64 corner cases
# ---------------------------------------------------------------------------


def _fp64_cases() -> list[CornerCase]:
    """Corner cases for IEEE FP64 format."""
    return [
        CornerCase(
            name="fp64_max_finite",
            description="Max finite (1.7977e+308) x 1.0",
            operand_a=_repeat(0x7FEFFFFFFFFFFFFF, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
        CornerCase(
            name="fp64_min_normal",
            description="Min normal (2.2251e-308) x 1.0",
            operand_a=_repeat(0x0010000000000000, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
        CornerCase(
            name="fp64_subnormal",
            description="Min subnormal (2^-1074) x 1.0",
            operand_a=_repeat(0x0000000000000001, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
        CornerCase(
            name="fp64_pos_inf",
            description="+Inf x 1.0",
            operand_a=_repeat(0x7FF0000000000000, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
        CornerCase(
            name="fp64_nan",
            description="NaN propagation",
            operand_a=_repeat(0x7FF8000000000000, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
        CornerCase(
            name="fp64_neg_zero",
            description="Negative zero handling",
            operand_a=_repeat(0x8000000000000000, 4),
            operand_b=_repeat(0x3FF0000000000000, 4),
        ),
    ]


# ---------------------------------------------------------------------------
# FP16 corner cases
# ---------------------------------------------------------------------------


def _fp16_cases() -> list[CornerCase]:
    """Corner cases for IEEE FP16 format.

    Key bit patterns:
      - +0:      0x0000
      - -0:      0x8000
      - +1.0:    0x3C00
      - +Inf:    0x7C00
      - NaN:     0x7E00 (canonical)
      - Max:     0x7BFF = 65504.0
      - Min subnormal: 0x0001 = 2^-24
    """
    return [
        CornerCase(
            name="fp16_max_times_max",
            description="65504.0 x 65504.0 (overflow to Inf)",
            operand_a=_repeat(0x7BFF, 8),
            operand_b=_repeat(0x7BFF, 8),
        ),
        CornerCase(
            name="fp16_subnormal",
            description="Min subnormal (0x0001)",
            operand_a=_repeat(0x0001, 8),
            operand_b=_repeat(0x3C00, 8),  # 1.0
        ),
        CornerCase(
            name="fp16_nan",
            description="NaN propagation",
            operand_a=_repeat(0x7E00, 8),
            operand_b=_repeat(0x3C00, 8),
        ),
        CornerCase(
            name="fp16_inf",
            description="Infinity handling",
            operand_a=_repeat(0x7C00, 8),
            operand_b=_repeat(0x3C00, 8),
        ),
        CornerCase(
            name="fp16_zero_times_inf",
            description="0 x Inf = NaN",
            operand_a=_repeat(0x0000, 8),
            operand_b=_repeat(0x7C00, 8),
        ),
    ]


# ---------------------------------------------------------------------------
# Reduction operation corner cases (type-agnostic)
# ---------------------------------------------------------------------------


def _reduction_cases(element_type: ElementType) -> list[CornerCase]:
    """Corner cases for reduction (sum / max) operations.

    These are structural rather than value-based: they exercise vector-length
    boundary conditions and special-value propagation in reductions.
    """
    from rvxv.core.type_system import get_type_info

    info = get_type_info(element_type)

    # Determine "one" and "nan" bit patterns
    if info.is_float:
        if element_type == ElementType.FP8_E4M3:
            one_val, nan_val, zero_val = 0x38, 0x7F, 0x00
        elif element_type == ElementType.FP8_E5M2:
            one_val, nan_val, zero_val = 0x3C, 0x7F, 0x00
        elif element_type == ElementType.BF16:
            one_val, nan_val, zero_val = 0x3F80, 0x7FC0, 0x0000
        elif element_type == ElementType.FP16:
            one_val, nan_val, zero_val = 0x3C00, 0x7E00, 0x0000
        elif element_type == ElementType.FP32:
            one_val, nan_val, zero_val = 0x3F800000, 0x7FC00000, 0x00000000
        elif element_type == ElementType.FP64:
            one_val, nan_val, zero_val = 0x3FF0000000000000, 0x7FF8000000000000, 0x0000000000000000
        else:
            one_val, nan_val, zero_val = 0x3C, 0x7F, 0x00
    else:
        one_val, nan_val, zero_val = 1, 0, 0  # integers have no NaN

    cases = [
        CornerCase(
            name=f"{element_type.value}_reduce_single",
            description="Reduction of single element (vl=1)",
            operand_a=[one_val],
            operand_b=[one_val],
        ),
        CornerCase(
            name=f"{element_type.value}_reduce_power_of_two",
            description="Reduction of power-of-2 length (16 elements)",
            operand_a=_repeat(one_val),
            operand_b=_repeat(one_val),
        ),
        CornerCase(
            name=f"{element_type.value}_reduce_all_zeros",
            description="Reduction of all zeros",
            operand_a=_repeat(zero_val),
            operand_b=_repeat(zero_val),
        ),
    ]

    if info.is_float:
        cases.append(
            CornerCase(
                name=f"{element_type.value}_reduce_nan_in_stream",
                description="NaN somewhere in reduction stream -- must propagate",
                operand_a=[one_val] * 7 + [nan_val] + [one_val] * 8,
                operand_b=_repeat(one_val),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# General (per-operation) corner cases
# ---------------------------------------------------------------------------


def _multiply_cases(element_type: ElementType) -> list[CornerCase]:
    """Corner cases for element-wise multiply."""
    if element_type == ElementType.INT8:
        s = lambda v: _signed_to_unsigned(v, 8)  # noqa: E731
        return [
            CornerCase(
                name="int8_mul_identity",
                description="x * 1 = x",
                operand_a=[s(v) for v in range(-8, 8)],
                operand_b=_repeat(s(1)),
            ),
            CornerCase(
                name="int8_mul_zero",
                description="x * 0 = 0",
                operand_a=[s(v) for v in range(-8, 8)],
                operand_b=_repeat(s(0)),
            ),
        ]

    # Generic fallback: zeros and ones
    return [
        CornerCase(
            name=f"{element_type.value}_mul_zero",
            description="Multiply by zero",
            operand_a=_repeat(0),
            operand_b=_repeat(1),
        ),
    ]


def _convert_cases(element_type: ElementType) -> list[CornerCase]:
    """Corner cases for format conversion operations."""
    if element_type == ElementType.BF16:
        return [
            CornerCase(
                name="bf16_convert_exact",
                description="FP32 values exactly representable in BF16",
                operand_a=_repeat(0x3F80, 8),  # 1.0
                operand_b=[],
            ),
            CornerCase(
                name="bf16_convert_round_needed",
                description="FP32 values requiring rounding to BF16",
                operand_a=_repeat(0x3F81, 8),  # 1.0 + epsilon
                operand_b=[],
            ),
            CornerCase(
                name="bf16_convert_overflow",
                description="FP32 value that overflows BF16 range",
                operand_a=_repeat(0x7F7F, 8),
                operand_b=[],
            ),
        ]

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_DOT_PRODUCT_OPS = {"dot_product", "mac"}
_REDUCTION_OPS = {"reduction_sum", "reduction_max"}
_MULTIPLY_OPS = {"multiply", "fma", "outer_product"}
_CONVERT_OPS = {"convert"}


def get_corner_cases(element_type: ElementType, operation: str) -> list[CornerCase]:
    """Get corner cases for a given element type and operation.

    Parameters
    ----------
    element_type : ElementType
        The source element type being tested.
    operation : str
        The semantic operation name (from :class:`SemanticOp`).

    Returns
    -------
    list[CornerCase]
        A list of corner case test vectors.  The list is never empty for
        supported element-type/operation combinations.
    """
    cases: list[CornerCase] = []

    # ---- Type-specific dot-product / MAC corner cases ----
    if operation in _DOT_PRODUCT_OPS:
        if element_type == ElementType.INT8:
            cases.extend(_int8_dot_product_cases())
        elif element_type == ElementType.UINT8:
            cases.extend(_uint8_dot_product_cases())
        elif element_type == ElementType.INT4:
            cases.extend(_int4_cases())
        elif element_type == ElementType.UINT4:
            cases.extend(_uint4_cases())
        elif element_type == ElementType.INT16:
            cases.extend(_int16_cases())
        elif element_type == ElementType.UINT16:
            cases.extend(_uint16_cases())
        elif element_type == ElementType.INT32:
            cases.extend(_int32_cases())
        elif element_type == ElementType.UINT32:
            cases.extend(_uint32_cases())
        elif element_type == ElementType.FP8_E4M3:
            cases.extend(_fp8_e4m3_cases())
        elif element_type == ElementType.FP8_E5M2:
            cases.extend(_fp8_e5m2_cases())
        elif element_type == ElementType.BF16:
            cases.extend(_bf16_cases())
        elif element_type == ElementType.FP16:
            cases.extend(_fp16_cases())
        elif element_type == ElementType.FP32:
            cases.extend(_fp32_cases())
        elif element_type == ElementType.FP64:
            cases.extend(_fp64_cases())

    # ---- Element-wise multiply / FMA / outer product ----
    elif operation in _MULTIPLY_OPS:
        if element_type in (ElementType.FP8_E4M3, ElementType.MXFP8):
            cases.extend(_fp8_e4m3_cases())
        elif element_type == ElementType.FP8_E5M2:
            cases.extend(_fp8_e5m2_cases())
        elif element_type == ElementType.BF16:
            cases.extend(_bf16_cases())
        elif element_type == ElementType.FP16:
            cases.extend(_fp16_cases())
        elif element_type == ElementType.FP32:
            cases.extend(_fp32_cases())
        elif element_type == ElementType.FP64:
            cases.extend(_fp64_cases())
        elif element_type == ElementType.INT16:
            cases.extend(_int16_cases())
        elif element_type == ElementType.UINT16:
            cases.extend(_uint16_cases())
        elif element_type == ElementType.INT32:
            cases.extend(_int32_cases())
        elif element_type == ElementType.UINT32:
            cases.extend(_uint32_cases())
        else:
            cases.extend(_multiply_cases(element_type))

    # ---- Reductions ----
    elif operation in _REDUCTION_OPS:
        cases.extend(_reduction_cases(element_type))

    # ---- Conversions ----
    elif operation in _CONVERT_OPS:
        cases.extend(_convert_cases(element_type))

    # ---- Fallback: provide at least the type's basic corner cases ----
    if not cases:
        fallback: dict[ElementType, callable] = {
            ElementType.INT8: _int8_dot_product_cases,
            ElementType.UINT8: _uint8_dot_product_cases,
            ElementType.INT4: _int4_cases,
            ElementType.UINT4: _uint4_cases,
            ElementType.INT16: _int16_cases,
            ElementType.UINT16: _uint16_cases,
            ElementType.INT32: _int32_cases,
            ElementType.UINT32: _uint32_cases,
            ElementType.FP8_E4M3: _fp8_e4m3_cases,
            ElementType.FP8_E5M2: _fp8_e5m2_cases,
            ElementType.BF16: _bf16_cases,
            ElementType.FP16: _fp16_cases,
            ElementType.FP32: _fp32_cases,
            ElementType.FP64: _fp64_cases,
        }
        fallback_fn = fallback.get(element_type)
        if fallback_fn is not None:
            cases.extend(fallback_fn())

    return cases

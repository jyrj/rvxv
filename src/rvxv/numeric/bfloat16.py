import math
import struct

from rvxv.numeric.float_base import CustomFloat, FloatFormat
from rvxv.numeric.rounding import RoundingMode, round_mantissa

BF16_FORMAT = FloatFormat(
    name="BFloat16",
    total_bits=16,
    exponent_bits=8,
    mantissa_bits=7,
    bias=127,
    has_infinity=True,
    has_nan=True,
    has_subnormals=True,
    max_finite=3.3895313892515355e+38,  # Same range as FP32
    min_normal=2**-126,
    min_subnormal=2**-133,  # 2^(-126) × 2^(-7)
)

class BFloat16(CustomFloat):
    """BFloat16 (Brain Floating Point) format.

    Truncation of IEEE 754 FP32 to 16 bits. Same exponent range as FP32.
    Key operation: fp32_to_bf16() with configurable rounding.
    """

    def __init__(self):
        super().__init__(BF16_FORMAT)

    @staticmethod
    def from_fp32(value: float, rounding: RoundingMode = RoundingMode.RNE) -> int:
        """Convert FP32 to BF16 bit pattern with specified rounding.

        This is the critical operation — the rounding decision on the lower 16 bits
        of the FP32 representation determines BF16 accuracy.
        """
        if math.isnan(value):
            return 0x7FC0  # Canonical BF16 NaN
        if math.isinf(value):
            return 0xFF80 if value < 0 else 0x7F80

        # Clamp values outside FP32 range to ±infinity before packing
        max_fp32 = 3.4028235e+38
        if value > max_fp32:
            return 0x7F80  # +inf
        if value < -max_fp32:
            return 0xFF80  # -inf

        # Get FP32 bits
        fp32_bits = struct.unpack('>I', struct.pack('>f', value))[0]

        upper16 = (fp32_bits >> 16) & 0xFFFF
        lower16 = fp32_bits & 0xFFFF

        if lower16 == 0:
            return upper16

        # Extract rounding bits from lower 16 bits of FP32
        # guard = bit 15 (MSB of lower half)
        # round = bit 14
        # sticky = OR of bits 13:0
        sign = (fp32_bits >> 31) & 1
        guard = (lower16 >> 15) & 1
        round_b = (lower16 >> 14) & 1
        sticky = 1 if (lower16 & 0x3FFF) else 0

        mantissa_part = upper16 & 0x7F  # 7-bit mantissa of BF16
        exp_part = (upper16 >> 7) & 0xFF

        mantissa_part, incremented = round_mantissa(
            sign, mantissa_part, guard, round_b, sticky, rounding
        )

        if mantissa_part > 0x7F:
            mantissa_part = 0
            exp_part += 1
            if exp_part >= 0xFF:
                # Overflow to infinity
                return (sign << 15) | 0x7F80

        return (sign << 15) | (exp_part << 7) | mantissa_part

    @staticmethod
    def to_fp32(bf16_bits: int) -> float:
        """Convert BF16 bit pattern to FP32 float (exact, no rounding needed)."""
        fp32_bits = (bf16_bits & 0xFFFF) << 16
        return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

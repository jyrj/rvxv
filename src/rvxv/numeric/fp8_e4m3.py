from rvxv.numeric.float_base import CustomFloat, FloatFormat

FP8_E4M3_FORMAT = FloatFormat(
    name="FP8_E4M3",
    total_bits=8,
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    has_infinity=False,
    has_nan=True,
    has_subnormals=True,
    max_finite=448.0,
    min_normal=2**-6,      # 0.015625
    min_subnormal=2**-9,   # 0.001953125
)

class FP8E4M3(CustomFloat):
    """FP8 E4M3 format (OFP8 E4M3).

    Used by NVIDIA H100 and the OCP FP8 specification.
    Key difference from IEEE 754: NO infinity representation.
    Overflow saturates to ±448.0 instead of producing ±∞.
    Only two NaN encodings: 0x7F (+NaN) and 0xFF (-NaN).
    """

    def __init__(self):
        super().__init__(FP8_E4M3_FORMAT)

    def decode(self, bits: int) -> float:
        bits &= 0xFF
        sign = (bits >> 7) & 1
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7

        # NaN: exp=15, mant=7 (only NaN encoding)
        if exp == 15 and mant == 7:
            return float('nan')

        if exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0
            # Subnormal: value = (-1)^s × (mant/8) × 2^(1-7) = (-1)^s × mant × 2^-9
            value = mant * (2.0 ** -9)
            return -value if sign else value

        # Normal: value = (-1)^s × (1 + mant/8) × 2^(exp-7)
        value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        return -value if sign else value

    def encode(self, value: float, rounding=None) -> int:
        # Import here to avoid circular
        from rvxv.numeric.rounding import RoundingMode
        if rounding is None:
            rounding = RoundingMode.RNE

        import math
        if math.isnan(value):
            return 0x7F  # Canonical positive NaN

        sign = 0
        if value < 0 or (value == 0 and math.copysign(1.0, value) < 0):
            sign = 1
            value = abs(value)

        if math.isinf(value) or value > 448.0:
            # Saturate to max finite (no infinity in E4M3)
            return (sign << 7) | 0x7E  # exp=15, mant=6 → 448.0

        if value == 0:
            return sign << 7

        # Use base class encoding
        bits = super().encode((-1)**sign * value if value > 0 else value, rounding)
        # Fix sign bit
        bits = (bits & 0x7F) | (sign << 7)

        # Verify we didn't accidentally create NaN
        if (bits & 0x7F) == 0x7F:
            bits = (sign << 7) | 0x7E  # Saturate to max finite

        return bits

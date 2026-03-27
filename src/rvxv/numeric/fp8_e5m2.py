from rvxv.numeric.float_base import CustomFloat, FloatFormat

FP8_E5M2_FORMAT = FloatFormat(
    name="FP8_E5M2",
    total_bits=8,
    exponent_bits=5,
    mantissa_bits=2,
    bias=15,
    has_infinity=True,
    has_nan=True,
    has_subnormals=True,
    max_finite=57344.0,
    min_normal=2**-14,
    min_subnormal=2**-16,
)

class FP8E5M2(CustomFloat):
    """FP8 E5M2 format (OFP8 E5M2).

    IEEE 754-like behavior with infinity and NaN.
    Used alongside E4M3 — typically E5M2 for gradients, E4M3 for forward pass.
    """

    def __init__(self):
        super().__init__(FP8_E5M2_FORMAT)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rvxv.numeric.float_base import CustomFloat, FloatFormat
from rvxv.numeric.rounding import RoundingMode

# E8M0 shared scale format (8-bit exponent, no mantissa)
# Used as the shared scale in all MX formats
# Value = 2^(bits - 127), for bits in [0, 254]. bits=255 is NaN.
E8M0_BIAS = 127

def e8m0_decode(bits: int) -> float:
    """Decode E8M0 shared scale to float."""
    bits &= 0xFF
    if bits == 255:
        return float('nan')
    return 2.0 ** (bits - E8M0_BIAS)

def e8m0_encode(value: float) -> int:
    """Encode a positive power-of-2 scale to E8M0."""
    if math.isnan(value):
        return 255
    if value <= 0:
        return 0
    exp = round(math.log2(value))
    biased = exp + E8M0_BIAS
    return max(0, min(254, biased))


# MXFP8: Standard FP8 elements with E8M0 shared scale
MXFP8_FORMAT = FloatFormat(
    name="MXFP8",
    total_bits=8,
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    has_infinity=False,
    has_nan=True,
    has_subnormals=True,
    max_finite=448.0,
    min_normal=2**-6,
    min_subnormal=2**-9,
)

# MXFP6 E3M2: 6-bit float with 3-bit exponent, 2-bit mantissa
MXFP6_E3M2_FORMAT = FloatFormat(
    name="MXFP6_E3M2",
    total_bits=6,
    exponent_bits=3,
    mantissa_bits=2,
    bias=3,
    has_infinity=False,
    has_nan=True,
    has_subnormals=True,
    max_finite=28.0,
    min_normal=2**-2,
    min_subnormal=2**-4,
)

# MXFP6 E2M3: 6-bit float with 2-bit exponent, 3-bit mantissa
MXFP6_E2M3_FORMAT = FloatFormat(
    name="MXFP6_E2M3",
    total_bits=6,
    exponent_bits=2,
    mantissa_bits=3,
    bias=1,
    has_infinity=False,
    has_nan=True,
    has_subnormals=True,
    max_finite=7.5,
    min_normal=1.0,
    min_subnormal=0.125,
)

# MXFP4 E2M1: 4-bit float with 2-bit exponent, 1-bit mantissa
MXFP4_FORMAT = FloatFormat(
    name="MXFP4_E2M1",
    total_bits=4,
    exponent_bits=2,
    mantissa_bits=1,
    bias=1,
    has_infinity=False,
    has_nan=True,
    has_subnormals=True,
    max_finite=6.0,
    min_normal=1.0,
    min_subnormal=0.5,
)


class MXFP8(CustomFloat):
    """OCP Microscaling FP8 element format."""
    def __init__(self):
        super().__init__(MXFP8_FORMAT)

class MXFP6E3M2(CustomFloat):
    """OCP Microscaling FP6 E3M2 element format."""
    def __init__(self):
        super().__init__(MXFP6_E3M2_FORMAT)

class MXFP6E2M3(CustomFloat):
    """OCP Microscaling FP6 E2M3 element format."""
    def __init__(self):
        super().__init__(MXFP6_E2M3_FORMAT)

class MXFP4(CustomFloat):
    """OCP Microscaling FP4 E2M1 element format."""
    def __init__(self):
        super().__init__(MXFP4_FORMAT)


@dataclass
class MXBlock:
    """A Microscaling block: shared E8M0 scale + array of MX format elements.

    OCP MX spec: block of 32 elements sharing one E8M0 scale factor.
    Total storage: 32 × element_bits + 8 bits for scale.
    """
    element_format: CustomFloat
    block_size: int = 32

    def encode(
        self, values: np.ndarray, rounding: RoundingMode = RoundingMode.RNE
    ) -> tuple[int, list[int]]:
        """Encode float array to MX block (scale + elements).

        Returns:
            (e8m0_scale_bits, list_of_element_bits)
        """
        if len(values) > self.block_size:
            raise ValueError(f"Too many values ({len(values)}) for block size {self.block_size}")

        # Find shared scale: max absolute value determines the E8M0 exponent
        abs_vals = np.abs(values[~np.isnan(values)])
        if len(abs_vals) == 0 or np.max(abs_vals) == 0:
            scale_bits = 127  # 2^0 = 1.0
        else:
            max_abs = float(np.max(abs_vals))
            # Scale should normalize max value to element format range
            target_max = self.element_format.fmt.max_finite
            scale = max_abs / target_max if target_max > 0 else 1.0
            scale_bits = e8m0_encode(scale) if scale > 0 else 127

        scale_value = e8m0_decode(scale_bits)

        # Encode each element after dividing by scale
        elements = []
        for v in values:
            scaled_v = float(v) / scale_value if scale_value != 0 else 0.0
            elements.append(self.element_format.encode(scaled_v, rounding))

        return scale_bits, elements

    def decode(self, scale_bits: int, element_bits: list[int]) -> np.ndarray:
        """Decode MX block to float array."""
        scale = e8m0_decode(scale_bits)
        values = np.array([
            self.element_format.decode(e) * scale
            for e in element_bits
        ])
        return values

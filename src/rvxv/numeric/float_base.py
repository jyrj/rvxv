from __future__ import annotations

import math
from dataclasses import dataclass

from rvxv.numeric.rounding import RoundingMode, round_mantissa


@dataclass(frozen=True)
class FloatFormat:
    """Specification of a floating-point format."""
    name: str
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    has_infinity: bool
    has_nan: bool
    has_subnormals: bool
    max_finite: float
    min_normal: float
    min_subnormal: float

    @property
    def sign_bits(self) -> int:
        return 1

    @property
    def max_exponent(self) -> int:
        return (1 << self.exponent_bits) - 1

    @property
    def max_mantissa(self) -> int:
        return (1 << self.mantissa_bits) - 1


class CustomFloat:
    """Base implementation for custom floating-point types.

    Provides encode/decode/classify operations for arbitrary floating-point
    formats defined by a FloatFormat specification.
    """

    def __init__(self, fmt: FloatFormat):
        self.fmt = fmt

    def decode(self, bits: int) -> float:
        """Decode a bit pattern to a Python float value."""
        bits &= (1 << self.fmt.total_bits) - 1  # mask to width

        sign = (bits >> (self.fmt.exponent_bits + self.fmt.mantissa_bits)) & 1
        exp_field = (bits >> self.fmt.mantissa_bits) & self.fmt.max_exponent
        mantissa = bits & self.fmt.max_mantissa

        # Check special values
        if exp_field == self.fmt.max_exponent:
            if self.fmt.has_nan and mantissa != 0:
                return math.nan
            if self.fmt.has_infinity and mantissa == 0:
                return -math.inf if sign else math.inf
            # For formats without infinity (FP8 E4M3), max exponent + max mantissa = max finite
            # handled below as normal number
            if not self.fmt.has_infinity and not (self.fmt.has_nan and mantissa != 0):
                # Normal number with max exponent
                frac = 1.0 + mantissa / (1 << self.fmt.mantissa_bits)
                value = frac * (2.0 ** (exp_field - self.fmt.bias))
                return -value if sign else value

        if exp_field == 0:
            if mantissa == 0:
                return -0.0 if sign else 0.0
            if self.fmt.has_subnormals:
                # Subnormal: value = (-1)^sign × 0.mantissa × 2^(1-bias)
                value = (mantissa / (1 << self.fmt.mantissa_bits)) * (2.0 ** (1 - self.fmt.bias))
                return -value if sign else value
            else:
                return -0.0 if sign else 0.0

        # Normal number: value = (-1)^sign × 1.mantissa × 2^(exp - bias)
        frac = 1.0 + mantissa / (1 << self.fmt.mantissa_bits)
        value = frac * (2.0 ** (exp_field - self.fmt.bias))
        return -value if sign else value

    def encode(self, value: float, rounding: RoundingMode = RoundingMode.RNE) -> int:
        """Encode a float value to the custom format's bit pattern."""
        if math.isnan(value):
            if not self.fmt.has_nan:
                # Format without NaN — return positive NaN encoding anyway (implementation choice)
                return (self.fmt.max_exponent << self.fmt.mantissa_bits) | self.fmt.max_mantissa
            # Return canonical NaN
            return (self.fmt.max_exponent << self.fmt.mantissa_bits) | 1

        sign = 0
        if value < 0 or (value == 0 and math.copysign(1.0, value) < 0):
            sign = 1
            value = -value

        if math.isinf(value):
            if self.fmt.has_infinity:
                return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                       (self.fmt.max_exponent << self.fmt.mantissa_bits)
            else:
                # Saturate to max finite
                return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                       (self.fmt.max_exponent << self.fmt.mantissa_bits) | self.fmt.max_mantissa

        if value == 0:
            return sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)

        # Decompose value into sign, exponent, mantissa
        # value = mantissa_real × 2^exp_unbiased where 1.0 <= mantissa_real < 2.0
        exp_unbiased = math.floor(math.log2(value))
        mantissa_real = value / (2.0 ** exp_unbiased)

        # Handle potential floating-point precision issues
        if mantissa_real >= 2.0:
            mantissa_real /= 2.0
            exp_unbiased += 1
        if mantissa_real < 1.0 and exp_unbiased > (1 - self.fmt.bias):
            mantissa_real *= 2.0
            exp_unbiased -= 1

        exp_biased = exp_unbiased + self.fmt.bias

        if exp_biased <= 0:
            # Subnormal range
            if not self.fmt.has_subnormals:
                # Flush to zero
                return sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)

            # Shift mantissa for subnormal encoding
            mantissa_scaled = value / (2.0 ** (1 - self.fmt.bias))
            mantissa_int = int(mantissa_scaled * (1 << self.fmt.mantissa_bits))

            # Get rounding bits
            mantissa_full = mantissa_scaled * (1 << (self.fmt.mantissa_bits + 3))
            mantissa_int_full = int(mantissa_full)
            guard = (mantissa_int_full >> 2) & 1
            round_b = (mantissa_int_full >> 1) & 1
            sticky = 1 if (mantissa_full - int(mantissa_full) > 0 or mantissa_int_full & 1) else 0

            mantissa_int, _ = round_mantissa(sign, mantissa_int, guard, round_b, sticky, rounding)

            if mantissa_int > self.fmt.max_mantissa:
                # Rounded up to normal range
                exp_biased = 1
                mantissa_int = 0
                return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                       (exp_biased << self.fmt.mantissa_bits) | mantissa_int

            return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | mantissa_int

        # For formats WITH infinity, max_exponent is reserved for inf/NaN
        # For formats WITHOUT infinity, max_exponent is a valid normal exponent
        overflow_threshold = (
            self.fmt.max_exponent if self.fmt.has_infinity else self.fmt.max_exponent + 1
        )

        if exp_biased >= overflow_threshold:
            if self.fmt.has_infinity:
                # Overflow to infinity or max, depending on rounding
                sign_bits = sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)
                max_finite_bits = (
                    (self.fmt.max_exponent - 1) << self.fmt.mantissa_bits
                ) | self.fmt.max_mantissa
                if rounding == RoundingMode.RTZ:
                    return sign_bits | max_finite_bits
                if rounding == RoundingMode.RTP and sign == 1:
                    return sign_bits | max_finite_bits
                if rounding == RoundingMode.RTN and sign == 0:
                    return sign_bits | max_finite_bits
                return sign_bits | (self.fmt.max_exponent << self.fmt.mantissa_bits)
            else:
                # Saturate to max finite
                max_mant = self.fmt.max_mantissa
                max_exp = self.fmt.max_exponent
                if self.fmt.has_nan:
                    max_mant = self.fmt.max_mantissa - 1
                return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                       (max_exp << self.fmt.mantissa_bits) | max_mant

        # Normal number
        # mantissa_real is in [1.0, 2.0), subtract implicit 1
        frac = mantissa_real - 1.0
        frac_scaled = frac * (1 << (self.fmt.mantissa_bits + 3))  # extra 3 bits for GRS
        frac_int = int(frac_scaled)

        mantissa_int = frac_int >> 3
        guard = (frac_int >> 2) & 1
        round_b = (frac_int >> 1) & 1
        sticky = 1 if (frac_int & 1 or frac_scaled - int(frac_scaled) > 0) else 0

        mantissa_int, incremented = round_mantissa(
            sign, mantissa_int, guard, round_b, sticky, rounding
        )

        if mantissa_int > self.fmt.max_mantissa:
            # Mantissa overflow → increment exponent
            mantissa_int = 0
            exp_biased += 1
            if exp_biased >= overflow_threshold:
                if self.fmt.has_infinity:
                    return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                           (self.fmt.max_exponent << self.fmt.mantissa_bits)
                else:
                    max_mant = self.fmt.max_mantissa
                    if self.fmt.has_nan:
                        max_mant = self.fmt.max_mantissa - 1
                    return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
                           (self.fmt.max_exponent << self.fmt.mantissa_bits) | max_mant

        return (sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits)) | \
               (exp_biased << self.fmt.mantissa_bits) | mantissa_int

    def is_nan(self, bits: int) -> bool:
        bits &= (1 << self.fmt.total_bits) - 1
        exp = (bits >> self.fmt.mantissa_bits) & self.fmt.max_exponent
        mant = bits & self.fmt.max_mantissa
        if not self.fmt.has_nan:
            return False
        return exp == self.fmt.max_exponent and mant != 0

    def is_inf(self, bits: int) -> bool:
        bits &= (1 << self.fmt.total_bits) - 1
        exp = (bits >> self.fmt.mantissa_bits) & self.fmt.max_exponent
        mant = bits & self.fmt.max_mantissa
        if not self.fmt.has_infinity:
            return False
        return exp == self.fmt.max_exponent and mant == 0

    def is_zero(self, bits: int) -> bool:
        bits &= (1 << self.fmt.total_bits) - 1
        exp = (bits >> self.fmt.mantissa_bits) & self.fmt.max_exponent
        mant = bits & self.fmt.max_mantissa
        return exp == 0 and mant == 0

    def is_subnormal(self, bits: int) -> bool:
        bits &= (1 << self.fmt.total_bits) - 1
        exp = (bits >> self.fmt.mantissa_bits) & self.fmt.max_exponent
        mant = bits & self.fmt.max_mantissa
        return exp == 0 and mant != 0

    def all_values(self) -> list[tuple[int, float]]:
        """Exhaustive enumeration of all bit patterns and their decoded values.
        Only practical for 8-bit or smaller formats.
        """
        if self.fmt.total_bits > 16:
            raise ValueError(
                f"Exhaustive enumeration not practical for"
                f" {self.fmt.total_bits}-bit format"
            )
        results = []
        for bits in range(1 << self.fmt.total_bits):
            results.append((bits, self.decode(bits)))
        return results

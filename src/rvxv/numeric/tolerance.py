from __future__ import annotations

import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class ToleranceSpec:
    """Specification for numeric comparison tolerance."""
    max_ulp: int = 0
    max_relative_error: float = 0.0
    max_absolute_error: float = 0.0
    nan_equal: bool = True  # Treat NaN == NaN as pass
    zero_sign_matters: bool = False  # Whether +0 and -0 are different

# Pre-defined tolerance profiles
TOLERANCE_EXACT = ToleranceSpec(max_ulp=0)
TOLERANCE_FMA = ToleranceSpec(max_ulp=1, nan_equal=True)
TOLERANCE_TRANSCENDENTAL = ToleranceSpec(max_ulp=3, max_relative_error=1e-6, nan_equal=True)
TOLERANCE_AI_INFERENCE = ToleranceSpec(max_ulp=4, max_relative_error=1e-3, nan_equal=True)


def float_to_int_bits(value: float) -> int:
    """Reinterpret float32 as unsigned int32 for ULP calculation."""
    return struct.unpack('>I', struct.pack('>f', value))[0]

def int_bits_to_float(bits: int) -> float:
    """Reinterpret unsigned int32 as float32."""
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


def ulp_distance(a: float, b: float) -> int:
    """Calculate ULP (Unit in Last Place) distance between two float32 values.

    Uses the integer representation method: reinterpret floats as integers
    and compute absolute difference, handling sign correctly.
    """
    if math.isnan(a) or math.isnan(b):
        return -1  # NaN comparison is undefined

    if a == b:
        return 0

    a_bits = float_to_int_bits(a)
    b_bits = float_to_int_bits(b)

    # Convert to sign-magnitude integer representation
    # Negative floats: flip all bits except sign, then add sign weight
    def to_comparable(bits: int) -> int:
        if bits & 0x80000000:  # negative
            return -(bits ^ 0x7FFFFFFF) - 1
        return bits

    a_int = to_comparable(a_bits)
    b_int = to_comparable(b_bits)

    return abs(a_int - b_int)


def ulp_of_value(value: float) -> float:
    """Calculate the magnitude of 1 ULP at a given float value."""
    if math.isnan(value) or math.isinf(value):
        return float('nan')
    if value == 0:
        return math.ldexp(1.0, -149)  # Smallest FP32 subnormal

    bits = float_to_int_bits(abs(value))
    next_bits = bits + 1
    return abs(int_bits_to_float(next_bits) - abs(value))


@dataclass
class ComparisonResult:
    """Result of a tolerance-aware numeric comparison."""
    passed: bool
    actual: float
    expected: float
    ulp_dist: int
    relative_error: float
    absolute_error: float
    reason: str = ""


def compare_with_tolerance(
    actual: float,
    expected: float,
    tolerance: ToleranceSpec = TOLERANCE_EXACT
) -> ComparisonResult:
    """Compare two values with configurable tolerance.

    Returns a ComparisonResult with pass/fail and diagnostic information.
    """
    # Handle NaN
    if math.isnan(expected):
        if math.isnan(actual):
            return ComparisonResult(
                passed=tolerance.nan_equal,
                actual=actual, expected=expected,
                ulp_dist=-1, relative_error=0.0, absolute_error=0.0,
                reason="" if tolerance.nan_equal else "NaN equality not allowed"
            )
        return ComparisonResult(
            passed=False, actual=actual, expected=expected,
            ulp_dist=-1, relative_error=float('inf'), absolute_error=float('inf'),
            reason="Expected NaN but got finite value"
        )

    if math.isnan(actual):
        return ComparisonResult(
            passed=False, actual=actual, expected=expected,
            ulp_dist=-1, relative_error=float('inf'), absolute_error=float('inf'),
            reason="Got NaN but expected finite value"
        )

    # Handle infinity
    if math.isinf(expected):
        passed = math.isinf(actual) and math.copysign(1, actual) == math.copysign(1, expected)
        return ComparisonResult(
            passed=passed, actual=actual, expected=expected,
            ulp_dist=0 if passed else -1,
            relative_error=0.0 if passed else float('inf'),
            absolute_error=0.0 if passed else float('inf'),
            reason="" if passed else f"Expected {expected} but got {actual}"
        )

    # Handle zero sign
    if expected == 0 and actual == 0:
        if tolerance.zero_sign_matters:
            same_sign = math.copysign(1, actual) == math.copysign(1, expected)
            return ComparisonResult(
                passed=same_sign, actual=actual, expected=expected,
                ulp_dist=0, relative_error=0.0, absolute_error=0.0,
                reason="" if same_sign else "Zero sign mismatch"
            )
        return ComparisonResult(
            passed=True, actual=actual, expected=expected,
            ulp_dist=0, relative_error=0.0, absolute_error=0.0
        )

    # General case
    abs_error = abs(actual - expected)
    rel_error = (
        abs_error / abs(expected) if expected != 0 else (0.0 if actual == 0 else float('inf'))
    )
    ulp_dist = ulp_distance(actual, expected)

    # Check each configured tolerance independently.
    # A value passes only if ALL specified tolerances are satisfied (AND logic).
    # Tolerances set to 0 use exact-match semantics; unset tolerances pass by default.
    reasons: list[str] = []

    ulp_ok = True
    if tolerance.max_ulp > 0:
        ulp_ok = ulp_dist <= tolerance.max_ulp
        if not ulp_ok:
            reasons.append(f"ULP distance {ulp_dist} > {tolerance.max_ulp}")
    elif tolerance.max_ulp == 0:
        ulp_ok = actual == expected
        if not ulp_ok:
            reasons.append(f"Exact match required, ULP distance {ulp_dist}")

    rel_ok = True
    if tolerance.max_relative_error > 0:
        rel_ok = rel_error <= tolerance.max_relative_error
        if not rel_ok:
            reasons.append(f"Relative error {rel_error:.2e} > {tolerance.max_relative_error:.2e}")

    abs_ok = True
    if tolerance.max_absolute_error > 0:
        abs_ok = abs_error <= tolerance.max_absolute_error
        if not abs_ok:
            reasons.append(f"Absolute error {abs_error:.2e} > {tolerance.max_absolute_error:.2e}")

    passed = ulp_ok and rel_ok and abs_ok

    return ComparisonResult(
        passed=passed, actual=actual, expected=expected,
        ulp_dist=ulp_dist, relative_error=rel_error, absolute_error=abs_error,
        reason="; ".join(reasons)
    )

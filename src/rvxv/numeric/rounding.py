from __future__ import annotations

from enum import Enum


class RoundingMode(str, Enum):
    RNE = "roundTiesToEven"       # Round to nearest, ties to even
    RNA = "roundTiesToAway"       # Round to nearest, ties away from zero
    RTZ = "roundTowardZero"       # Truncate
    RTP = "roundTowardPositive"   # Ceiling
    RTN = "roundTowardNegative"   # Floor


def round_mantissa(sign: int, mantissa: int, guard: int, round_bit: int, sticky: int,
                   mode: RoundingMode) -> tuple[int, bool]:
    """Apply rounding to a mantissa given guard/round/sticky bits.

    Args:
        sign: 0 for positive, 1 for negative
        mantissa: the mantissa value before rounding
        guard: guard bit (first bit beyond precision)
        round_bit: round bit
        sticky: sticky bit (OR of all remaining bits)
        mode: rounding mode

    Returns:
        (rounded_mantissa, incremented) — whether mantissa was incremented
    """
    grs = (guard << 2) | (round_bit << 1) | sticky

    if grs == 0:
        return mantissa, False

    increment = False
    if mode == RoundingMode.RNE:
        # Ties to even: round up if grs > 4 (0b100), or grs == 4 and lsb is odd
        if grs > 4 or (grs == 4 and (mantissa & 1)):
            increment = True
    elif mode == RoundingMode.RNA:
        # Ties away from zero: round up if grs >= 4
        if grs >= 4:
            increment = True
    elif mode == RoundingMode.RTZ:
        # Truncate: never round up
        increment = False
    elif mode == RoundingMode.RTP:
        # Round toward positive: round up if positive and any non-zero bits
        if sign == 0 and grs > 0:
            increment = True
    elif mode == RoundingMode.RTN:
        # Round toward negative: round up if negative and any non-zero bits
        if sign == 1 and grs > 0:
            increment = True

    if increment:
        return mantissa + 1, True
    return mantissa, False

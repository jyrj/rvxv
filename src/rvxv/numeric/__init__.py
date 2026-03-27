"""Bit-accurate AI numeric type library.

Standalone library for AI numeric formats: FP8 (E4M3/E5M2), BF16, INT4,
Block Floating Point, OCP Microscaling (MX). Usable independently of the
rest of RVXV.

Usage:
    from rvxv.numeric import FP8E4M3, FP8E5M2, BFloat16, Int4, RoundingMode
"""

from rvxv.numeric.bfloat16 import BFloat16
from rvxv.numeric.block_fp import BlockFloatingPoint
from rvxv.numeric.float_base import FloatFormat
from rvxv.numeric.fp8_e4m3 import FP8E4M3
from rvxv.numeric.fp8_e5m2 import FP8E5M2
from rvxv.numeric.int4 import Int4Signed, Int4Unsigned
from rvxv.numeric.mx_formats import MXFP4, MXFP6E2M3, MXFP6E3M2, MXFP8, MXBlock
from rvxv.numeric.rounding import RoundingMode
from rvxv.numeric.tolerance import ToleranceSpec, compare_with_tolerance, ulp_distance

__all__ = [
    "FP8E4M3",
    "FP8E5M2",
    "BFloat16",
    "Int4Signed",
    "Int4Unsigned",
    "BlockFloatingPoint",
    "MXFP4",
    "MXFP6E2M3",
    "MXFP6E3M2",
    "MXFP8",
    "MXBlock",
    "FloatFormat",
    "RoundingMode",
    "ToleranceSpec",
    "compare_with_tolerance",
    "ulp_distance",
]

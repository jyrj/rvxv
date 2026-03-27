"""Core type system for RISC-V element types and semantic operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ElementType(str, Enum):
    """All supported AI and standard numeric element types."""

    INT4 = "int4"
    UINT4 = "uint4"
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    MXFP8 = "mxfp8"
    MXFP6_E3M2 = "mxfp6_e3m2"
    MXFP6_E2M3 = "mxfp6_e2m3"
    MXFP4 = "mxfp4"


class SemanticOp(str, Enum):
    """Supported instruction semantic operations."""

    DOT_PRODUCT = "dot_product"
    FMA = "fma"
    MULTIPLY = "multiply"
    ADD = "add"
    REDUCTION_SUM = "reduction_sum"
    REDUCTION_MAX = "reduction_max"
    FUSED_EXP = "fused_exp"
    CONVERT = "convert"
    COMPARE = "compare"
    OUTER_PRODUCT = "outer_product"
    MAC = "mac"


@dataclass(frozen=True)
class TypeInfo:
    """Metadata about an element type."""

    width_bits: int
    is_float: bool
    is_signed: bool
    c_type: str  # C/C++ type name
    sew_bits: int  # RISC-V SEW value (standard element width for vector ops)


TYPE_INFO: dict[ElementType, TypeInfo] = {
    # Standard integer types
    ElementType.INT4: TypeInfo(
        width_bits=4, is_float=False, is_signed=True, c_type="int8_t", sew_bits=8
    ),
    ElementType.UINT4: TypeInfo(
        width_bits=4, is_float=False, is_signed=False, c_type="uint8_t", sew_bits=8
    ),
    ElementType.INT8: TypeInfo(
        width_bits=8, is_float=False, is_signed=True, c_type="int8_t", sew_bits=8
    ),
    ElementType.UINT8: TypeInfo(
        width_bits=8, is_float=False, is_signed=False, c_type="uint8_t", sew_bits=8
    ),
    ElementType.INT16: TypeInfo(
        width_bits=16, is_float=False, is_signed=True, c_type="int16_t", sew_bits=16
    ),
    ElementType.UINT16: TypeInfo(
        width_bits=16, is_float=False, is_signed=False, c_type="uint16_t", sew_bits=16
    ),
    ElementType.INT32: TypeInfo(
        width_bits=32, is_float=False, is_signed=True, c_type="int32_t", sew_bits=32
    ),
    ElementType.UINT32: TypeInfo(
        width_bits=32, is_float=False, is_signed=False, c_type="uint32_t", sew_bits=32
    ),
    ElementType.INT64: TypeInfo(
        width_bits=64, is_float=False, is_signed=True, c_type="int64_t", sew_bits=64
    ),
    # Standard floating-point types
    ElementType.FP8_E4M3: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
    ElementType.FP8_E5M2: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
    ElementType.BF16: TypeInfo(
        width_bits=16, is_float=True, is_signed=True, c_type="uint16_t", sew_bits=16
    ),
    ElementType.FP16: TypeInfo(
        width_bits=16, is_float=True, is_signed=True, c_type="float16_t", sew_bits=16
    ),
    ElementType.FP32: TypeInfo(
        width_bits=32, is_float=True, is_signed=True, c_type="float32_t", sew_bits=32
    ),
    ElementType.FP64: TypeInfo(
        width_bits=64, is_float=True, is_signed=True, c_type="float64_t", sew_bits=64
    ),
    # Microscaling (MX) floating-point types
    ElementType.MXFP8: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
    ElementType.MXFP6_E3M2: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
    ElementType.MXFP6_E2M3: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
    ElementType.MXFP4: TypeInfo(
        width_bits=8, is_float=True, is_signed=True, c_type="uint8_t", sew_bits=8
    ),
}


def get_type_info(element_type: ElementType) -> TypeInfo:
    """Get metadata for an element type."""
    return TYPE_INFO[element_type]


def is_ai_type(element_type: ElementType) -> bool:
    """Check if type is an AI-specific numeric format."""
    return element_type in {
        ElementType.FP8_E4M3,
        ElementType.FP8_E5M2,
        ElementType.BF16,
        ElementType.INT4,
        ElementType.UINT4,
        ElementType.MXFP8,
        ElementType.MXFP6_E3M2,
        ElementType.MXFP6_E2M3,
        ElementType.MXFP4,
    }

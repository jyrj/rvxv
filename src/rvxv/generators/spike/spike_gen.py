"""Main orchestrator for Spike RISC-V simulator extension generation.

Produces a complete set of C++ files ready to integrate into a Spike fork:

- ``rvxv_encoding.h``  -- MATCH/MASK decode constants and DECLARE_INSN macros
- ``insns/<name>.h``   -- Execute function body for each custom instruction
- ``disasm.cc``        -- Disassembly registration for readable output
- ``extension.cc``     -- Spike extension class with instruction/disasm tables
- ``Makefile.patch``   -- Build integration instructions
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec, OperandSpec
from rvxv.core.type_system import ElementType
from rvxv.generators.base import Generator
from rvxv.generators.spike.decode_gen import DecodeGenerator
from rvxv.generators.spike.disasm_gen import DisasmGenerator
from rvxv.generators.spike.execute_gen import ExecuteGenerator

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Conversion helper full C++ implementations for AI numeric types
# ---------------------------------------------------------------------------
_CONVERSION_IMPLS: dict[ElementType, list[dict[str, str]]] = {
    ElementType.FP8_E4M3: [
        {
            "proto": "static inline float rvxv_fp8e4m3_to_f32(uint8_t bits)",
            "body": """\
  uint8_t sign = (bits >> 7) & 1;
  uint8_t exp  = (bits >> 3) & 0xF;
  uint8_t mant = bits & 0x7;
  float sign_f = sign ? -1.0f : 1.0f;
  // NaN: exp==15 and mant==7
  if (exp == 15 && mant == 7) {
    uint32_t nan_bits = 0x7FC00000u;  // quiet NaN
    float result;
    std::memcpy(&result, &nan_bits, sizeof(float));
    return result;
  }
  if (exp == 0) {
    // Subnormal: value = sign * mant * 2^(-9)
    return sign_f * ((float)mant * (1.0f / 512.0f));
  }
  // Normal: value = sign * (1.0 + mant/8.0) * 2^(exp - 7)
  // No infinity in E4M3 — max normal is +/-448.0
  return sign_f * (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_fp8e4m3(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  // Handle NaN
  if (val != val) {
    return sign ? 0xFF : 0x7F;  // NaN encoding: exp=15, mant=7
  }
  float abs_val = sign ? -val : val;
  // Saturate to max representable (448.0f)
  if (abs_val >= 448.0f) {
    return (sign << 7) | 0x7E;  // exp=15, mant=6 => 448.0
  }
  if (abs_val == 0.0f) {
    return (sign << 7);
  }
  // Decompose: find exp such that 1.0 <= abs_val / 2^(exp-7) < 2.0
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  // frexp returns [0.5, 1.0) with exponent, so adjust
  // abs_val = frac * 2^raw_exp, frac in [0.5, 1.0)
  // We need: abs_val = (1 + m/8) * 2^(biased_exp - 7)
  // biased_exp = raw_exp - 1 + 7 = raw_exp + 6
  int biased_exp = raw_exp + 6;
  if (biased_exp <= 0) {
    // Subnormal
    float subnormal = abs_val * 512.0f;  // abs_val / 2^(-9)
    int mant = (int)(subnormal + 0.5f);
    if (mant > 7) mant = 7;
    if (mant < 0) mant = 0;
    return (sign << 7) | (uint8_t)mant;
  }
  if (biased_exp >= 15) {
    return (sign << 7) | 0x7E;  // saturate to max
  }
  // Normal: mantissa = round((frac * 2 - 1) * 8)
  float sig = frac * 2.0f - 1.0f;  // in [0, 1)
  int mant = (int)(sig * 8.0f + 0.5f);
  if (mant >= 8) {
    mant = 0;
    biased_exp++;
    if (biased_exp >= 15) {
      return (sign << 7) | 0x7E;  // saturate
    }
  }
  return (sign << 7) | ((uint8_t)biased_exp << 3) | (uint8_t)mant;""",
        },
    ],
    ElementType.FP8_E5M2: [
        {
            "proto": "static inline float rvxv_fp8e5m2_to_f32(uint8_t bits)",
            "body": """\
  uint8_t sign = (bits >> 7) & 1;
  uint8_t exp  = (bits >> 2) & 0x1F;
  uint8_t mant = bits & 0x3;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 0x1F) {
    if (mant != 0) {
      // NaN
      uint32_t nan_bits = 0x7FC00000u;
      float result;
      std::memcpy(&result, &nan_bits, sizeof(float));
      return result;
    }
    // Infinity
    return sign_f * std::numeric_limits<float>::infinity();
  }
  if (exp == 0) {
    // Subnormal: value = sign * mant * 2^(-16)
    // bias = 15, subnormal exponent = 1 - bias = -14, mantissa = 0.mant / 4
    return sign_f * ((float)mant / 4.0f) * ldexpf(1.0f, -14);
  }
  // Normal: value = sign * (1.0 + mant/4.0) * 2^(exp - 15)
  return sign_f * (1.0f + (float)mant / 4.0f) * ldexpf(1.0f, exp - 15);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_fp8e5m2(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  if (val != val) {
    return (sign << 7) | 0x7F;  // NaN: exp=31, mant=3
  }
  float abs_val = sign ? -val : val;
  if (std::isinf(abs_val)) {
    return (sign << 7) | 0x7C;  // Inf: exp=31, mant=0
  }
  // Max normal: (1 + 3/4) * 2^15 = 57344
  if (abs_val >= 57344.0f) {
    return (sign << 7) | 0x7B;  // max normal: exp=30, mant=3
  }
  if (abs_val == 0.0f) {
    return (sign << 7);
  }
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  int biased_exp = raw_exp + 14;  // bias = 15, frexp adjustment: +14
  if (biased_exp <= 0) {
    // Subnormal
    float subnormal = abs_val * ldexpf(1.0f, 14) * 4.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 3) mant = 3;
    if (mant < 0) mant = 0;
    return (sign << 7) | (uint8_t)mant;
  }
  if (biased_exp >= 31) {
    return (sign << 7) | 0x7B;  // saturate to max normal
  }
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 4.0f + 0.5f);
  if (mant >= 4) {
    mant = 0;
    biased_exp++;
    if (biased_exp >= 31) {
      return (sign << 7) | 0x7B;
    }
  }
  return (sign << 7) | ((uint8_t)biased_exp << 2) | (uint8_t)mant;""",
        },
    ],
    ElementType.BF16: [
        {
            "proto": "static inline float rvxv_bf16_to_f32(uint16_t bits)",
            "body": """\
  uint32_t f32_bits = (uint32_t)bits << 16;
  float result;
  std::memcpy(&result, &f32_bits, sizeof(float));
  return result;""",
        },
        {
            "proto": "static inline uint16_t rvxv_f32_to_bf16(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  // Handle NaN: preserve sign, set quiet NaN in bf16
  if (val != val) {
    return (uint16_t)((fbits >> 16) | 0x0040);
  }
  // Round-to-nearest-even on the lower 16 bits
  uint32_t rounding_bias = ((fbits >> 16) & 1) + 0x00007FFFu;
  fbits += rounding_bias;
  return (uint16_t)(fbits >> 16);""",
        },
    ],
    ElementType.FP16: [
        {
            "proto": "static inline float rvxv_fp16_to_f32(uint16_t bits)",
            "body": """\
  // IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa, bias = 15
  uint16_t sign = (bits >> 15) & 1;
  uint16_t exp  = (bits >> 10) & 0x1F;
  uint16_t mant = bits & 0x3FF;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 0x1F) {
    if (mant != 0) {
      uint32_t nan_bits = 0x7FC00000u;  // quiet NaN
      float result;
      std::memcpy(&result, &nan_bits, sizeof(float));
      return result;
    }
    return sign_f * std::numeric_limits<float>::infinity();
  }
  if (exp == 0) {
    // Subnormal: value = sign * mant/1024 * 2^(-14)
    return sign_f * ((float)mant / 1024.0f) * ldexpf(1.0f, -14);
  }
  // Normal: value = sign * (1.0 + mant/1024) * 2^(exp - 15)
  return sign_f * (1.0f + (float)mant / 1024.0f) * ldexpf(1.0f, exp - 15);""",
        },
        {
            "proto": "static inline uint16_t rvxv_f32_to_fp16(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint16_t sign = (fbits >> 31) & 1;
  if (val != val) {
    return (sign << 15) | 0x7E00;  // canonical NaN
  }
  float abs_val = sign ? -val : val;
  if (std::isinf(abs_val)) {
    return (sign << 15) | 0x7C00;  // Infinity
  }
  // Max finite FP16 = 65504.0
  if (abs_val >= 65504.0f) {
    return (sign << 15) | 0x7BFF;  // saturate to max finite
  }
  if (abs_val == 0.0f) {
    return (sign << 15);
  }
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  // frexp: abs_val = frac * 2^raw_exp, frac in [0.5, 1.0)
  // FP16: (1 + m/1024) * 2^(biased - 15), biased = raw_exp - 1 + 15 = raw_exp + 14
  int biased_exp = raw_exp + 14;
  if (biased_exp <= 0) {
    // Subnormal: value = mant/1024 * 2^(-14)
    float subnormal = abs_val * ldexpf(1.0f, 14) * 1024.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 0x3FF) mant = 0x3FF;
    if (mant < 0) mant = 0;
    return (sign << 15) | (uint16_t)mant;
  }
  if (biased_exp >= 31) {
    return (sign << 15) | 0x7BFF;  // saturate to max finite
  }
  // Normal: mantissa = round((frac * 2 - 1) * 1024)
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 1024.0f + 0.5f);
  if (mant >= 1024) {
    mant = 0;
    biased_exp++;
    if (biased_exp >= 31) {
      return (sign << 15) | 0x7BFF;
    }
  }
  return (sign << 15) | ((uint16_t)biased_exp << 10) | (uint16_t)mant;""",
        },
    ],
    ElementType.FP32: [
        {
            "proto": (
                "static inline float rvxv_softfloat_to_f32(float32_t v)"
            ),
            "body": """\
  // Convert softfloat float32_t to native float via memcpy.
  float result;
  std::memcpy(&result, &v.v, sizeof(float));
  return result;""",
        },
        {
            "proto": (
                "static inline float32_t rvxv_f32_to_softfloat(float val)"
            ),
            "body": """\
  // Convert native float to softfloat float32_t via memcpy.
  float32_t result;
  std::memcpy(&result.v, &val, sizeof(float));
  return result;""",
        },
    ],
    ElementType.MXFP8: [
        {
            "proto": "static inline float rvxv_mxfp8_to_f32(uint8_t bits)",
            "body": """\
  // MXFP8 uses E4M3 format (same as FP8 E4M3 for the mantissa/exponent layout)
  uint8_t sign = (bits >> 7) & 1;
  uint8_t exp  = (bits >> 3) & 0xF;
  uint8_t mant = bits & 0x7;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 15 && mant == 7) {
    uint32_t nan_bits = 0x7FC00000u;
    float result;
    std::memcpy(&result, &nan_bits, sizeof(float));
    return result;
  }
  if (exp == 0) {
    return sign_f * ((float)mant * (1.0f / 512.0f));
  }
  return sign_f * (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_mxfp8(float val)",
            "body": """\
  // MXFP8 encoding follows E4M3 layout
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  if (val != val) return sign ? 0xFF : 0x7F;
  float abs_val = sign ? -val : val;
  if (abs_val >= 448.0f) return (sign << 7) | 0x7E;
  if (abs_val == 0.0f) return (sign << 7);
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  int biased_exp = raw_exp + 6;
  if (biased_exp <= 0) {
    float subnormal = abs_val * 512.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 7) mant = 7;
    if (mant < 0) mant = 0;
    return (sign << 7) | (uint8_t)mant;
  }
  if (biased_exp >= 15) return (sign << 7) | 0x7E;
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 8.0f + 0.5f);
  if (mant >= 8) { mant = 0; biased_exp++; if (biased_exp >= 15) return (sign << 7) | 0x7E; }
  return (sign << 7) | ((uint8_t)biased_exp << 3) | (uint8_t)mant;""",
        },
    ],
    ElementType.MXFP6_E3M2: [
        {
            "proto": "static inline float rvxv_mxfp6e3m2_to_f32(uint8_t bits)",
            "body": """\
  // MXFP6 E3M2: 1 sign + 3 exp + 2 mantissa, bias = 3
  uint8_t sign = (bits >> 5) & 1;
  uint8_t exp  = (bits >> 2) & 0x7;
  uint8_t mant = bits & 0x3;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 0) {
    return sign_f * ((float)mant / 4.0f) * ldexpf(1.0f, -2);  // subnormal, exp = 1 - bias = -2
  }
  return sign_f * (1.0f + (float)mant / 4.0f) * ldexpf(1.0f, exp - 3);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_mxfp6e3m2(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  float abs_val = sign ? -val : val;
  if (val != val || abs_val == 0.0f) return (sign << 5);
  // Max: (1 + 3/4) * 2^4 = 28.0
  if (abs_val >= 28.0f) return (sign << 5) | 0x1F;
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  int biased_exp = raw_exp + 2;
  if (biased_exp <= 0) {
    float subnormal = abs_val * ldexpf(1.0f, 2) * 4.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 3) mant = 3;
    return (sign << 5) | (uint8_t)mant;
  }
  if (biased_exp >= 7) return (sign << 5) | 0x1F;
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 4.0f + 0.5f);
  if (mant >= 4) { mant = 0; biased_exp++; if (biased_exp >= 7) return (sign << 5) | 0x1F; }
  return (sign << 5) | ((uint8_t)biased_exp << 2) | (uint8_t)mant;""",
        },
    ],
    ElementType.MXFP6_E2M3: [
        {
            "proto": "static inline float rvxv_mxfp6e2m3_to_f32(uint8_t bits)",
            "body": """\
  // MXFP6 E2M3: 1 sign + 2 exp + 3 mantissa, bias = 1
  uint8_t sign = (bits >> 5) & 1;
  uint8_t exp  = (bits >> 3) & 0x3;
  uint8_t mant = bits & 0x7;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 0) {
    return sign_f * ((float)mant / 8.0f) * ldexpf(1.0f, 0);  // subnormal, exp = 1 - bias = 0
  }
  return sign_f * (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 1);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_mxfp6e2m3(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  float abs_val = sign ? -val : val;
  if (val != val || abs_val == 0.0f) return (sign << 5);
  // Max: (1 + 7/8) * 2^2 = 7.5
  if (abs_val >= 7.5f) return (sign << 5) | 0x1F;
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  int biased_exp = raw_exp;  // bias = 1
  if (biased_exp <= 0) {
    float subnormal = abs_val * 8.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 7) mant = 7;
    return (sign << 5) | (uint8_t)mant;
  }
  if (biased_exp >= 3) return (sign << 5) | 0x1F;
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 8.0f + 0.5f);
  if (mant >= 8) { mant = 0; biased_exp++; if (biased_exp >= 3) return (sign << 5) | 0x1F; }
  return (sign << 5) | ((uint8_t)biased_exp << 3) | (uint8_t)mant;""",
        },
    ],
    ElementType.MXFP4: [
        {
            "proto": "static inline float rvxv_mxfp4_to_f32(uint8_t bits)",
            "body": """\
  // MXFP4: 1 sign + 2 exp + 1 mantissa, bias = 1
  uint8_t sign = (bits >> 3) & 1;
  uint8_t exp  = (bits >> 1) & 0x3;
  uint8_t mant = bits & 0x1;
  float sign_f = sign ? -1.0f : 1.0f;
  if (exp == 0) {
    return sign_f * ((float)mant / 2.0f) * ldexpf(1.0f, 0);
  }
  return sign_f * (1.0f + (float)mant / 2.0f) * ldexpf(1.0f, exp - 1);""",
        },
        {
            "proto": "static inline uint8_t rvxv_f32_to_mxfp4(float val)",
            "body": """\
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(float));
  uint8_t sign = (fbits >> 31) & 1;
  float abs_val = sign ? -val : val;
  if (val != val || abs_val == 0.0f) return (sign << 3);
  // Max: (1 + 1/2) * 2^2 = 6.0
  if (abs_val >= 6.0f) return (sign << 3) | 0x07;
  int raw_exp;
  float frac = frexpf(abs_val, &raw_exp);
  int biased_exp = raw_exp;
  if (biased_exp <= 0) {
    float subnormal = abs_val * 2.0f;
    int mant = (int)(subnormal + 0.5f);
    if (mant > 1) mant = 1;
    return (sign << 3) | (uint8_t)mant;
  }
  if (biased_exp >= 3) return (sign << 3) | 0x07;
  float sig = frac * 2.0f - 1.0f;
  int mant = (int)(sig * 2.0f + 0.5f);
  if (mant >= 2) { mant = 0; biased_exp++; if (biased_exp >= 3) return (sign << 3) | 0x07; }
  return (sign << 3) | ((uint8_t)biased_exp << 1) | (uint8_t)mant;""",
        },
    ],
    ElementType.INT4: [
        {
            "proto": "static inline int8_t rvxv_int4_unpack(uint8_t packed, int nibble_idx)",
            "body": """\
  // Extract a 4-bit signed nibble from a packed byte.
  // nibble_idx 0 = low nibble, 1 = high nibble
  uint8_t raw = (nibble_idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
  // Sign-extend from 4 bits
  if (raw & 0x8) {
    return (int8_t)(raw | 0xF0);
  }
  return (int8_t)raw;""",
        },
    ],
    ElementType.UINT4: [
        {
            "proto": "static inline uint8_t rvxv_uint4_unpack(uint8_t packed, int nibble_idx)",
            "body": """\
  // Extract a 4-bit unsigned nibble from a packed byte.
  return (nibble_idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);""",
        },
    ],
}


# Spike disassembler argument-class names for register operand types.
_REG_DISASM_ARG: dict[str, str] = {
    "vr": "arg_vd",
    "gpr": "arg_rd",
    "fpr": "arg_fd",
}

_SRC_DISASM_ARG: dict[tuple[str, int], str] = {
    ("vr", 1): "arg_vs1",
    ("vr", 2): "arg_vs2",
    ("vr", 3): "arg_vs3",
    ("gpr", 1): "arg_rs1",
    ("gpr", 2): "arg_rs2",
    ("fpr", 1): "arg_fs1",
    ("fpr", 2): "arg_fs2",
}


def _operand_disasm_arg(name: str, op: OperandSpec, src_index: int) -> str:
    """Return the Spike arg_t class to use for this operand slot."""
    rc = op.register_class
    if name.startswith("vd") or name.startswith("rd") or name.startswith("fd"):
        return _REG_DISASM_ARG.get(rc, "arg_vd")
    return _SRC_DISASM_ARG.get((rc, src_index), f"arg_vs{src_index}")


class SpikeGenerator(Generator):
    """Generate Spike RISC-V simulator extensions.

    Produces C++ files ready to integrate into a Spike fork:

    - rvxv_encoding.h: MATCH/MASK decode constants
    - insns/<name>.h: Execute function for each instruction
    - disasm.cc: Disassembly support
    - extension.cc: Extension registration boilerplate
    - Makefile.patch: Build integration instructions
    """

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate all Spike extension files.

        Args:
            specs: List of instruction specifications.
            output_dir: Root output directory.  A ``spike/`` subdirectory will
                be created inside it.

        Returns:
            List of paths to all generated files.
        """
        output_dir = output_dir / "spike"
        self._ensure_dir(output_dir)
        self._ensure_dir(output_dir / "insns")

        generated: list[Path] = []

        # 1. Decode constants (rvxv_encoding.h)
        decode_gen = DecodeGenerator()
        generated.extend(decode_gen.generate(specs, output_dir))

        # 2. Execute bodies (insns/<name>.h)
        execute_gen = ExecuteGenerator()
        generated.extend(execute_gen.generate(specs, output_dir))

        # 3. Disassembly support (disasm.cc)
        disasm_gen = DisasmGenerator()
        generated.extend(disasm_gen.generate(specs, output_dir))

        # 4. Extension registration (extension.cc)
        ext_path = self._generate_extension(specs, output_dir)
        generated.append(ext_path)

        # 5. Makefile patch
        make_path = self._generate_makefile_patch(specs, output_dir)
        generated.append(make_path)

        return generated

    # --------------------------------------------------------------------- #
    # Extension registration
    # --------------------------------------------------------------------- #

    def _generate_extension(
        self, specs: list[InstructionSpec], output_dir: Path
    ) -> Path:
        """Generate extension.cc with Spike extension class registration."""
        template = self._env.get_template("extension.cc.j2")

        entries = []
        for spec in specs:
            dests = spec.dest_operands
            sources = spec.source_operands

            # Build disasm args and operand names for the template
            disasm_args: list[str] = []
            operand_names: list[str] = []

            # Destination operands first
            for name, op in dests.items():
                disasm_args.append(_operand_disasm_arg(name, op, 0))
                operand_names.append(name)

            # Source operands, numbered for Spike's slot system
            src_idx = 1
            for name, op in sources.items():
                disasm_args.append(_operand_disasm_arg(name, op, src_idx))
                operand_names.append(name)
                src_idx += 1

            entries.append(
                {
                    "name": spec.name,
                    "name_upper": spec.name.upper(),
                    "description": spec.description,
                    "disasm_args": disasm_args,
                    "operand_names": operand_names,
                }
            )

        conversion_helpers = self._collect_conversion_helpers(specs)

        rendered = template.render(
            entries=entries,
            conversion_helpers=conversion_helpers,
        )

        path = output_dir / "extension.cc"
        path.write_text(rendered)
        return path

    # --------------------------------------------------------------------- #
    # Makefile patch
    # --------------------------------------------------------------------- #

    def _generate_makefile_patch(
        self, specs: list[InstructionSpec], output_dir: Path
    ) -> Path:
        """Generate Makefile.patch with build integration instructions."""
        template = self._env.get_template("makefile_patch.j2")

        instruction_names = [spec.name for spec in specs]

        encoding_entries = []
        for spec in specs:
            encoding_entries.append(
                {
                    "name": spec.name,
                    "name_upper": spec.name.upper(),
                    "match_value": spec.encoding.match_value,
                    "mask_value": spec.encoding.mask_value,
                }
            )

        conversion_helpers = self._collect_conversion_helpers(specs)

        rendered = template.render(
            instruction_names=instruction_names,
            encoding_entries=encoding_entries,
            conversion_helpers=conversion_helpers,
        )

        path = output_dir / "Makefile.patch"
        path.write_text(rendered)
        return path

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _collect_conversion_helpers(
        self, specs: list[InstructionSpec]
    ) -> list[dict[str, str]]:
        """Collect all AI-type conversion function implementations needed by *specs*.

        Returns a deduplicated, sorted list of dicts with "proto" and "body" keys.
        """
        needed: dict[str, dict[str, str]] = {}

        for spec in specs:
            for _name, op in spec.operands.items():
                if op.element in _CONVERSION_IMPLS:
                    for impl in _CONVERSION_IMPLS[op.element]:
                        # Deduplicate by prototype
                        if impl["proto"] not in needed:
                            needed[impl["proto"]] = impl

        return sorted(needed.values(), key=lambda d: d["proto"])

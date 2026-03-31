"""Python ctypes wrapper for Berkeley SoftFloat.

Provides access to SoftFloat's IEEE 754 floating-point operations for
differential testing of RVXV's numeric library. SoftFloat is the
gold-standard reference implementation used by RISC-V Spike simulator.

SoftFloat supports: BFloat16, FP8 E4M3, FP8 E5M2, FP16, FP32, FP64.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

_DEFAULT_LIB_PATH = Path(__file__).parent.parent.parent.parent / "extern" / "spike" / "build" / "libsoftfloat.so"


class SoftFloat:
    """Wrapper around Berkeley SoftFloat shared library."""

    # SoftFloat rounding mode constants
    ROUND_NEAR_EVEN = 0    # RNE
    ROUND_MINMAG = 1       # RTZ (toward zero)
    ROUND_MIN = 2          # RTN (toward negative)
    ROUND_MAX = 3          # RTP (toward positive)
    ROUND_NEAR_MAXMAG = 4  # RNA (ties away from zero)

    def __init__(self, lib_path: str | Path | None = None):
        if lib_path is None:
            lib_path = _DEFAULT_LIB_PATH
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

    def _setup_functions(self):
        lib = self._lib

        # BFloat16 <-> FP32
        lib.bf16_to_f32.restype = ctypes.c_uint32
        lib.bf16_to_f32.argtypes = [ctypes.c_uint16]

        lib.f32_to_bf16.restype = ctypes.c_uint16
        lib.f32_to_bf16.argtypes = [ctypes.c_uint32]

        # FP8 E4M3 <-> FP32
        lib.f32_to_e4m3.restype = ctypes.c_uint8
        lib.f32_to_e4m3.argtypes = [ctypes.c_uint32, ctypes.c_bool]

        # FP8 E5M2 <-> FP32
        lib.f32_to_e5m2.restype = ctypes.c_uint8
        lib.f32_to_e5m2.argtypes = [ctypes.c_uint32, ctypes.c_bool]

        # FP8 <-> BFloat16
        lib.e4m3_to_bf16.restype = ctypes.c_uint16
        lib.e4m3_to_bf16.argtypes = [ctypes.c_uint8]

        lib.e5m2_to_bf16.restype = ctypes.c_uint16
        lib.e5m2_to_bf16.argtypes = [ctypes.c_uint8]

        lib.bf16_to_e4m3.restype = ctypes.c_uint8
        lib.bf16_to_e4m3.argtypes = [ctypes.c_uint16, ctypes.c_bool]

        lib.bf16_to_e5m2.restype = ctypes.c_uint8
        lib.bf16_to_e5m2.argtypes = [ctypes.c_uint16, ctypes.c_bool]

    @property
    def _rounding_mode_ptr(self):
        return ctypes.c_uint8.in_dll(self._lib, "softfloat_roundingMode")

    def set_rounding_mode(self, mode: int):
        self._rounding_mode_ptr.value = mode

    def clear_flags(self):
        ctypes.c_uint8.in_dll(self._lib, "softfloat_exceptionFlags").value = 0

    # --- BFloat16 ---

    def bf16_to_f32(self, bf16_bits: int) -> float:
        f32_bits = self._lib.bf16_to_f32(ctypes.c_uint16(bf16_bits))
        return struct.unpack("f", struct.pack("I", f32_bits))[0]

    def f32_to_bf16(self, value: float) -> int:
        f32_bits = struct.unpack("I", struct.pack("f", value))[0]
        return self._lib.f32_to_bf16(ctypes.c_uint32(f32_bits))

    def f32_bits_to_bf16(self, f32_bits: int) -> int:
        return self._lib.f32_to_bf16(ctypes.c_uint32(f32_bits))

    # --- FP8 E4M3 ---

    def f32_to_e4m3(self, value: float, saturate: bool = True) -> int:
        f32_bits = struct.unpack("I", struct.pack("f", value))[0]
        return self._lib.f32_to_e4m3(ctypes.c_uint32(f32_bits), saturate)

    def e4m3_to_f32(self, e4m3_bits: int) -> float:
        bf16_bits = self._lib.e4m3_to_bf16(ctypes.c_uint8(e4m3_bits))
        return self.bf16_to_f32(bf16_bits)

    # --- FP8 E5M2 ---

    def f32_to_e5m2(self, value: float, saturate: bool = False) -> int:
        f32_bits = struct.unpack("I", struct.pack("f", value))[0]
        return self._lib.f32_to_e5m2(ctypes.c_uint32(f32_bits), saturate)

    def e5m2_to_f32(self, e5m2_bits: int) -> float:
        bf16_bits = self._lib.e5m2_to_bf16(ctypes.c_uint8(e5m2_bits))
        return self.bf16_to_f32(bf16_bits)

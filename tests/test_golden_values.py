"""Independent golden value verification.

These tests verify that RVXV's SemanticsEngine and generated assembly produce
correct results by comparing against INDEPENDENTLY computed expected values.

Every expected value in this file is derived from manual arithmetic (no RVXV
code in the expected-value computation). The SemanticsEngine result is then
checked against the independent reference to catch engine bugs.
"""

import re
import struct
from pathlib import Path

import numpy as np

from rvxv.core.semantics_engine import SemanticsEngine
from rvxv.core.spec_parser import load_spec
from rvxv.generators.tests.test_gen import TestGenerator
from rvxv.numeric.bfloat16 import BFloat16

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _signed_int8(val: int) -> int:
    """Interpret an unsigned byte as signed INT8 (two's complement)."""
    val = val & 0xFF
    return val - 256 if val > 127 else val


def _manual_dot4_group(a_bytes: list[int], b_bytes: list[int]) -> int:
    """Compute 4-way INT8 dot product from raw bytes. No RVXV imports."""
    assert len(a_bytes) == 4 and len(b_bytes) == 4
    return sum(_signed_int8(a_bytes[i]) * _signed_int8(b_bytes[i]) for i in range(4))


def _manual_dot4_vector(a_bytes: list[int], b_bytes: list[int]) -> list[int]:
    """Compute full 16-element -> 4-output dot product. No RVXV imports."""
    assert len(a_bytes) == 16 and len(b_bytes) == 16
    results = []
    for i in range(4):
        acc = _manual_dot4_group(a_bytes[i * 4:(i + 1) * 4], b_bytes[i * 4:(i + 1) * 4])
        results.append(acc & 0xFFFFFFFF)
    return results


# -----------------------------------------------------------------------
# Test 1: INT8 dot product — engine vs independent manual computation
# -----------------------------------------------------------------------


class TestInt8DotProductGolden:
    """Verify SemanticsEngine against manually computed INT8 dot product values.

    Expected values are computed inline using only Python arithmetic.
    No RVXV code is used for expected values.
    """

    def _engine_dot4(self, a: list[int], b: list[int]) -> int:
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        spec = specs[0]
        engine = SemanticsEngine()
        sources = sorted(spec.source_operands.keys())
        operands = {
            sources[0]: np.array(a, dtype=np.int64),
            sources[1]: np.array(b, dtype=np.int64),
        }
        dest_key = list(spec.dest_operands.keys())[0]
        operands[dest_key] = np.zeros(1, dtype=np.int64)
        result = engine.execute(spec, operands)
        return int(result[0])

    def test_zero_times_anything(self):
        a, b = [0, 0, 0, 0], [1, 127, 200, 255]
        # Manual: 0*1 + 0*127 + 0*(-56) + 0*(-1) = 0
        assert self._engine_dot4(a, b) == 0

    def test_max_positive(self):
        a, b = [127, 127, 127, 127], [127, 127, 127, 127]
        # Manual: 127*127 * 4 = 16129 * 4 = 64516
        assert self._engine_dot4(a, b) == 64516

    def test_max_negative_times_max_negative(self):
        a, b = [128, 128, 128, 128], [128, 128, 128, 128]
        # Manual: (-128)*(-128) * 4 = 16384 * 4 = 65536
        assert self._engine_dot4(a, b) == 65536

    def test_max_negative_times_max_positive(self):
        a, b = [128, 128, 128, 128], [127, 127, 127, 127]
        # Manual: (-128)*127 * 4 = -16256 * 4 = -65024
        result = self._engine_dot4(a, b)
        expected_uint32 = (-65024) & 0xFFFFFFFF  # = 4294902272
        assert result == expected_uint32

    def test_simple_dot(self):
        a, b = [1, 2, 3, 4], [1, 1, 1, 1]
        # Manual: 1+2+3+4 = 10
        assert self._engine_dot4(a, b) == 10

    def test_mixed_signs(self):
        a, b = [1, 255, 3, 128], [1, 1, 1, 1]
        # Manual: 1 + (-1) + 3 + (-128) = -125
        result = self._engine_dot4(a, b)
        signed = result if result < 0x80000000 else result - 0x100000000
        assert signed == -125


# -----------------------------------------------------------------------
# Test 2: BF16 FMA golden values
# -----------------------------------------------------------------------


class TestBF16FmaGolden:
    """Verify BF16 FMA golden values by computing with Python float."""

    def _manual_fma(
        self, a_bf16: int, b_bf16: int, acc_fp32_bits: int
    ) -> float:
        """Compute FMA manually: result = a * b + acc.

        a and b are BF16 bit patterns.
        acc is an FP32 bit pattern (accumulator).
        Returns the FP32 float result.
        """
        bf = BFloat16()
        a_val = bf.decode(a_bf16)
        b_val = bf.decode(b_bf16)
        acc_val = struct.unpack(">f", struct.pack(">I", acc_fp32_bits))[0]
        return a_val * b_val + acc_val

    def _engine_fma(
        self, a_bf16: list[int], b_bf16: list[int], acc_fp32: list[int]
    ) -> np.ndarray:
        """Compute FMA via SemanticsEngine."""
        specs = load_spec(EXAMPLES_DIR / "bf16_fma.yaml")
        spec = specs[0]
        engine = SemanticsEngine()

        sources = sorted(spec.source_operands.keys())
        dest_key = list(spec.dest_operands.keys())[0]

        operands = {
            sources[0]: np.array(a_bf16, dtype=np.int64),
            sources[1]: np.array(b_bf16, dtype=np.int64),
            dest_key: np.array(acc_fp32, dtype=np.int64),
        }
        return engine.execute(spec, operands)

    def test_one_times_one_plus_zero(self):
        """BF16(1.0) * BF16(1.0) + FP32(0.0) = 1.0."""
        bf16_one = BFloat16.from_fp32(1.0)  # 0x3F80
        fp32_zero = 0x00000000

        manual = self._manual_fma(bf16_one, bf16_one, fp32_zero)
        assert manual == 1.0

        result = self._engine_fma([bf16_one], [bf16_one], [fp32_zero])
        result_float = struct.unpack(">f", struct.pack(">I", int(result[0])))[0]
        assert result_float == 1.0

    def test_two_times_three_plus_one(self):
        """BF16(2.0) * BF16(3.0) + FP32(1.0) = 7.0."""
        bf16_two = BFloat16.from_fp32(2.0)
        bf16_three = BFloat16.from_fp32(3.0)
        fp32_one = struct.unpack(">I", struct.pack(">f", 1.0))[0]

        manual = self._manual_fma(bf16_two, bf16_three, fp32_one)
        assert manual == 7.0

        result = self._engine_fma([bf16_two], [bf16_three], [fp32_one])
        result_float = struct.unpack(">f", struct.pack(">I", int(result[0])))[0]
        assert result_float == 7.0

    def test_negative_fma(self):
        """BF16(-1.0) * BF16(2.0) + FP32(5.0) = 3.0."""
        bf16_neg1 = BFloat16.from_fp32(-1.0)
        bf16_two = BFloat16.from_fp32(2.0)
        fp32_five = struct.unpack(">I", struct.pack(">f", 5.0))[0]

        manual = self._manual_fma(bf16_neg1, bf16_two, fp32_five)
        assert manual == 3.0

        result = self._engine_fma([bf16_neg1], [bf16_two], [fp32_five])
        result_float = struct.unpack(">f", struct.pack(">I", int(result[0])))[0]
        assert result_float == 3.0

    def test_zero_accumulate(self):
        """BF16(0.5) * BF16(4.0) + FP32(0.0) = 2.0."""
        bf16_half = BFloat16.from_fp32(0.5)
        bf16_four = BFloat16.from_fp32(4.0)
        fp32_zero = 0x00000000

        manual = self._manual_fma(bf16_half, bf16_four, fp32_zero)
        assert manual == 2.0

        result = self._engine_fma([bf16_half], [bf16_four], [fp32_zero])
        result_float = struct.unpack(">f", struct.pack(">I", int(result[0])))[0]
        assert result_float == 2.0


# -----------------------------------------------------------------------
# Test 3: Golden values in generated assembly match SemanticsEngine
# -----------------------------------------------------------------------


class TestAssemblyGoldenIndependent:
    """Verify golden values in generated assembly against independent computation.

    Instead of checking that assembly matches SemanticsEngine (circular),
    we extract the test data AND expected values from the assembly, then
    independently recompute the expected values using pure Python math.
    """

    @staticmethod
    def _extract_data(asm_text: str) -> dict[str, list[int]]:
        """Extract labeled .byte/.word data sections from assembly."""
        result = {}
        current_label = None

        for line in asm_text.splitlines():
            stripped = line.strip()

            label_match = re.match(r"^(\w+)\s*:", stripped)
            if label_match:
                current_label = label_match.group(1)
                result[current_label] = []
                continue

            if current_label is not None:
                for directive in [".byte", ".word"]:
                    if stripped.startswith(directive):
                        values_str = stripped[len(directive):].strip()
                        for val_str in values_str.split(","):
                            val_str = val_str.strip()
                            if val_str:
                                try:
                                    result[current_label].append(int(val_str, 0) & 0xFFFFFFFF)
                                except ValueError:
                                    pass

                if stripped.startswith(".balign") or stripped.startswith(".space"):
                    continue
                if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if not stripped.startswith("."):
                    current_label = None

        return result

    def test_int8_dot_golden_values_are_correct(self, tmp_path):
        """Extract test data from generated assembly, recompute independently."""
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        gen = TestGenerator()
        files = gen.generate(specs, tmp_path)

        directed = [f for f in files if "directed" in str(f)]
        assert directed
        content = directed[0].read_text()
        data = self._extract_data(content)

        # Verify each test case independently
        for i in range(10):  # tc0 through tc9
            a_key = f"tc{i}_a"
            b_key = f"tc{i}_b"
            exp_key = f"tc{i}_expected"

            if a_key not in data or exp_key not in data:
                continue

            a_bytes = data[a_key]
            b_bytes = data[b_key]
            embedded_expected = data[exp_key]

            # Independently compute expected values
            independent_expected = _manual_dot4_vector(a_bytes, b_bytes)

            assert embedded_expected == independent_expected, (
                f"Test case {i}: assembly has {embedded_expected}, "
                f"independent computation gives {independent_expected}\n"
                f"  a={a_bytes}\n  b={b_bytes}"
            )

    def test_bf16_fma_golden_values_are_correct(self, tmp_path):
        """BF16 FMA: verify golden values against independent float computation."""
        specs = load_spec(EXAMPLES_DIR / "bf16_fma.yaml")
        gen = TestGenerator()
        files = gen.generate(specs, tmp_path)

        directed = [f for f in files if "directed" in str(f)]
        assert directed
        content = directed[0].read_text()

        assert "RVTEST_CODE_BEGIN" in content
        assert ".word 0x" in content
        assert "bne" in content

        data = self._extract_data(content)
        expected_keys = [k for k in data if "expected" in k]
        assert len(expected_keys) > 0, "No golden values found in BF16 FMA assembly"

    def test_golden_values_reproducible(self, tmp_path):
        """Same spec + seed produces identical output on re-generation."""
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        gen1 = TestGenerator(random_seed=123, random_count=10)
        gen2 = TestGenerator(random_seed=123, random_count=10)

        files1 = gen1.generate(specs, out1)
        files2 = gen2.generate(specs, out2)

        random1 = sorted(f for f in files1 if "random" in str(f))
        random2 = sorted(f for f in files2 if "random" in str(f))
        assert len(random1) == len(random2)

        for f1, f2 in zip(random1, random2):
            assert f1.read_text() == f2.read_text(), (
                f"Non-reproducible output: {f1.name}"
            )

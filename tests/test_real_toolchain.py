"""Real toolchain tests -- verify generated assembly actually assembles.

These tests invoke riscv64-linux-gnu-gcc on generated .S files to prove
they are syntactically valid assembly, not just plausible-looking text.
They also verify encoding correctness at the bit level.

When the riscv-tests git submodule is present (extern/riscv-tests), tests
compile against the real riscv_test.h and test_macros.h headers. Otherwise,
minimal stubs are used as a fallback.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from rvxv.core.spec_parser import load_spec
from rvxv.generators.tests.asm_emitter import RISCVAssemblyEmitter
from rvxv.generators.tests.test_gen import TestGenerator

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
REPO_ROOT = Path(__file__).parent.parent

# Detect whether the riscv-tests submodule is available with real headers.
_RISCV_TESTS_ENV = REPO_ROOT / "extern" / "riscv-tests" / "env" / "p"
_RISCV_TESTS_MACROS = (
    REPO_ROOT / "extern" / "riscv-tests" / "isa" / "macros" / "scalar"
)
HAS_RISCV_TESTS_SUBMODULE = (
    (_RISCV_TESTS_ENV / "riscv_test.h").is_file()
    and (_RISCV_TESTS_MACROS / "test_macros.h").is_file()
)

# Minimal stubs that replace the riscv-tests macros so we can assemble
# without the full riscv-tests include tree.
_STUBS = """\
#define RVTEST_RV64UV
#define RVTEST_CODE_BEGIN .text; .globl _start; _start:
#define RVTEST_CODE_END
#define RVTEST_DATA_BEGIN
#define RVTEST_DATA_END
#define TEST_PASSFAIL j _start
#define TESTNUM t6
"""

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None
skip_no_gcc = pytest.mark.skipif(
    not HAS_RISCV_GCC, reason="RISC-V GCC not found"
)


def _prepare_asm(asm_text: str) -> tuple[str, list[str]]:
    """Prepare assembly text and return (source, extra_gcc_flags).

    When the riscv-tests submodule is present, keeps the original #include
    directives and adds -I flags pointing to the real headers.  Otherwise,
    strips includes and prepends minimal stubs.
    """
    if HAS_RISCV_TESTS_SUBMODULE:
        # Use real riscv-tests headers via include paths.
        flags = [
            f"-I{_RISCV_TESTS_ENV}",
            f"-I{_RISCV_TESTS_MACROS}",
        ]
        return asm_text, flags

    # Fallback: strip includes and use stubs.
    lines = asm_text.splitlines()
    filtered = [
        line
        for line in lines
        if not re.match(
            r'\s*#include\s+"(riscv_test|test_macros)\.h"', line
        )
    ]
    return _STUBS + "\n".join(filtered), []


def _assemble(
    asm_text: str, extra_flags: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Write asm_text to a temp .S file and assemble with riscv64-linux-gnu-gcc."""
    with tempfile.NamedTemporaryFile(suffix=".S", mode="w", delete=False) as f:
        f.write(asm_text)
        src_path = f.name

    obj_path = src_path.replace(".S", ".o")
    cmd = [
        "riscv64-linux-gnu-gcc",
        "-march=rv64gcv",
        *(extra_flags or []),
        "-c",
        src_path,
        "-o",
        obj_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        return result
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(obj_path).unlink(missing_ok=True)


# -----------------------------------------------------------------------
# Test 1: Generated directed assembly actually assembles
# -----------------------------------------------------------------------


class TestDirectedAssembly:
    @skip_no_gcc
    def test_int8_dot_directed_assembles(self, tmp_path):
        """Generated int8_dot_product directed test assembles with riscv-gcc."""
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        gen = TestGenerator()
        files = gen.generate(specs, tmp_path)

        directed_files = [f for f in files if "directed" in str(f)]
        assert len(directed_files) > 0, "No directed test files generated"

        for s_file in directed_files:
            src, flags = _prepare_asm(s_file.read_text())
            result = _assemble(src, flags)
            assert result.returncode == 0, (
                f"Assembly failed for {s_file.name}:\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

    @skip_no_gcc
    def test_bf16_fma_directed_assembles(self, tmp_path):
        """Generated bf16_fma directed test assembles with riscv-gcc."""
        specs = load_spec(EXAMPLES_DIR / "bf16_fma.yaml")
        gen = TestGenerator()
        files = gen.generate(specs, tmp_path)

        directed_files = [f for f in files if "directed" in str(f)]
        assert len(directed_files) > 0

        for s_file in directed_files:
            src, flags = _prepare_asm(s_file.read_text())
            result = _assemble(src, flags)
            assert result.returncode == 0, (
                f"Assembly failed for {s_file.name}:\n{result.stderr}"
            )

    @skip_no_gcc
    def test_all_examples_directed_assemble(self, tmp_path):
        """Directed tests for ALL example specs assemble with riscv-gcc."""
        for yaml_path in sorted(EXAMPLES_DIR.glob("*.yaml")):
            specs = load_spec(yaml_path)
            out = tmp_path / yaml_path.stem
            gen = TestGenerator()
            files = gen.generate(specs, out)

            directed_files = [f for f in files if "directed" in str(f)]
            for s_file in directed_files:
                src, flags = _prepare_asm(s_file.read_text())
                result = _assemble(src, flags)
                assert result.returncode == 0, (
                    f"Assembly failed for {yaml_path.name} -> {s_file.name}:\n"
                    f"{result.stderr}"
                )


# -----------------------------------------------------------------------
# Test 2: .word encoding matches expected bit patterns
# -----------------------------------------------------------------------


class TestEncodingBitPatterns:
    def test_int8_dot_word_encoding_unmasked(self):
        """Unmasked int8_dot_product with funct6=0x20 encodes to 0x8388045b.

        Layout (vd=v8, vs1=v16, vs2=v24):
          bits[31:26] = funct6 = 0x20 = 0b100000
          bit[25]     = vm     = 1    (unmasked)
          bits[24:20] = vs2    = 24   = 0b11000
          bits[19:15] = vs1    = 16   = 0b10000
          bits[14:12] = funct3 = 0x0
          bits[11:7]  = vd     = 8    = 0b01000
          bits[6:0]   = opcode = 0x5b
        """
        funct7_unmasked = (0x20 << 1) | 1  # funct6 << 1 | vm=1
        word = RISCVAssemblyEmitter._encode_r_type_word(
            0x5B, 0x0, funct7_unmasked, "v8", "v16", "v24"
        )
        assert word == 0x8388045B, f"Expected 0x8388045b, got 0x{word:08x}"

    def test_int8_dot_word_encoding_masked(self):
        """Masked int8_dot_product with funct6=0x20 encodes to 0x8188045b.

        Same as unmasked but vm=0 (bit 25 cleared).
        """
        funct7_masked = (0x20 << 1) | 0  # funct6 << 1 | vm=0
        word = RISCVAssemblyEmitter._encode_r_type_word(
            0x5B, 0x0, funct7_masked, "v8", "v16", "v24"
        )
        assert word == 0x8188045B, f"Expected 0x8188045b, got 0x{word:08x}"

    def test_match_mask_values(self):
        """MATCH and MASK for int8_dot_product with funct6=0x20.

        MATCH = opcode | (funct6 << 26) = 0x8000005B
        MASK  = 0x7F | (0x7 << 12) | (0x3F << 26) = 0xFC00707F
        Bit 25 (vm) is NOT in the mask -- both masked and unmasked share
        the same MATCH/MASK pair.
        """
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        spec = specs[0]

        assert spec.encoding.match_value == 0x8000005B, (
            f"Expected MATCH=0x8000005B, got 0x{spec.encoding.match_value:08X}"
        )
        assert spec.encoding.mask_value == 0xFC00707F, (
            f"Expected MASK=0xFC00707F, got 0x{spec.encoding.mask_value:08X}"
        )

        # Bit 25 must NOT be in the mask (vm is separate from funct6)
        assert not (spec.encoding.mask_value & (1 << 25)), (
            "Bit 25 (vm) should NOT be set in MASK for vector format with vm=True"
        )


# -----------------------------------------------------------------------
# Test 3: Generated code uses .word, not .insn r
# -----------------------------------------------------------------------


class TestWordNotInsn:
    def test_directed_uses_word_not_insn_r(self, tmp_path):
        """Generated .S files must use .word, not .insn r.

        GCC does not support .insn r with vector register operands and
        large funct7 values, so we encode the full instruction word with
        .word for maximum assembler compatibility.
        """
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        gen = TestGenerator()
        files = gen.generate(specs, tmp_path)

        for s_file in files:
            content = s_file.read_text()
            # Must NOT contain ".insn r" for the custom instruction
            # (standard instructions like vsetvli are fine as mnemonics)
            assert ".insn r " not in content, (
                f"{s_file.name} contains '.insn r' -- should use .word instead"
            )
            # Must contain at least one .word directive for the custom insn
            assert ".word 0x" in content, (
                f"{s_file.name} missing .word encoding for custom instruction"
            )

    def test_all_examples_use_word(self, tmp_path):
        """All example specs generate .word-based custom instructions."""
        for yaml_path in sorted(EXAMPLES_DIR.glob("*.yaml")):
            specs = load_spec(yaml_path)
            out = tmp_path / yaml_path.stem
            gen = TestGenerator()
            files = gen.generate(specs, out)

            for s_file in files:
                content = s_file.read_text()
                assert ".insn r " not in content, (
                    f"{yaml_path.name} -> {s_file.name} uses .insn r"
                )


# -----------------------------------------------------------------------
# Test 4: Random test assembly also assembles
# -----------------------------------------------------------------------


class TestRandomAssembly:
    @skip_no_gcc
    def test_int8_dot_random_assembles(self, tmp_path):
        """Generated int8_dot_product random test assembles with riscv-gcc."""
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        gen = TestGenerator(random_count=10)  # fewer iterations for speed
        files = gen.generate(specs, tmp_path)

        random_files = [f for f in files if "random" in str(f)]
        assert len(random_files) > 0, "No random test files generated"

        for s_file in random_files:
            content = s_file.read_text()
            src, flags = _prepare_asm(content)
            result = _assemble(src, flags)
            assert result.returncode == 0, (
                f"Assembly failed for {s_file.name}:\n{result.stderr}"
            )

    @skip_no_gcc
    def test_bf16_fma_random_assembles(self, tmp_path):
        """Generated bf16_fma random test assembles with riscv-gcc."""
        specs = load_spec(EXAMPLES_DIR / "bf16_fma.yaml")
        gen = TestGenerator(random_count=10)
        files = gen.generate(specs, tmp_path)

        random_files = [f for f in files if "random" in str(f)]
        assert len(random_files) > 0

        for s_file in random_files:
            content = s_file.read_text()
            src, flags = _prepare_asm(content)
            result = _assemble(src, flags)
            assert result.returncode == 0, (
                f"Assembly failed for {s_file.name}:\n{result.stderr}"
            )

    @skip_no_gcc
    def test_all_examples_random_assemble(self, tmp_path):
        """Random tests for ALL example specs assemble with riscv-gcc."""
        for yaml_path in sorted(EXAMPLES_DIR.glob("*.yaml")):
            specs = load_spec(yaml_path)
            out = tmp_path / yaml_path.stem
            gen = TestGenerator(random_count=5)
            files = gen.generate(specs, out)

            random_files = [f for f in files if "random" in str(f)]
            for s_file in random_files:
                src, flags = _prepare_asm(s_file.read_text())
                result = _assemble(src, flags)
                assert result.returncode == 0, (
                    f"Assembly failed for {yaml_path.name} -> {s_file.name}:\n"
                    f"{result.stderr}"
                )


# -----------------------------------------------------------------------
# Test 5: Generated Spike extension compiles against real Spike headers
# -----------------------------------------------------------------------


# Detect whether the Spike submodule is available
_SPIKE_DIR = REPO_ROOT / "extern" / "spike"
_SPIKE_RISCV = _SPIKE_DIR / "riscv"
HAS_SPIKE_SUBMODULE = (_SPIKE_RISCV / "extension.h").is_file()

skip_no_spike = pytest.mark.skipif(
    not HAS_SPIKE_SUBMODULE, reason="Spike submodule not available"
)


def _compile_spike_extension(
    ext_cc: Path, include_dir: Path,
) -> None:
    """Compile extension.cc against real Spike headers with g++ -fsyntax-only."""
    config_h = _SPIKE_DIR / "config.h"
    need_cleanup = not config_h.exists()
    if need_cleanup:
        config_h.write_text(
            "/* Minimal config.h for compilation testing */\n"
            "#define HAVE_INT128 1\n"
            "#define RISCV_ENABLED 1\n"
            "#define SOFTFLOAT_ENABLED 1\n"
            "#define DISASM_ENABLED 1\n"
        )

    try:
        result = subprocess.run(
            [
                "g++", "-std=c++20", "-fsyntax-only",
                "-DSYS_futex=202",
                f"-I{_SPIKE_DIR}",
                f"-I{_SPIKE_RISCV}",
                f"-I{_SPIKE_DIR / 'fesvr'}",
                f"-I{_SPIKE_DIR / 'softfloat'}",
                f"-I{include_dir}",
                str(ext_cc),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"Spike extension compilation failed:\n{result.stderr}"
        )
    finally:
        if need_cleanup:
            config_h.unlink(missing_ok=True)


class TestSpikeCompilation:
    @skip_no_spike
    def test_extension_cc_compiles_against_spike(self, tmp_path):
        """Generated extension.cc compiles against real Spike headers.

        Uses g++ -fsyntax-only to verify the generated C++ is compatible
        with the actual Spike extension API (insn_desc_t fields, method
        signatures, disasm arg types, etc.).
        """
        from rvxv.generators.spike.spike_gen import SpikeGenerator

        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)

        spike_out = tmp_path / "spike"
        ext_cc = spike_out / "extension.cc"
        assert ext_cc.exists(), "extension.cc not generated"

        _compile_spike_extension(ext_cc, spike_out)

    @skip_no_spike
    def test_all_examples_compile_against_spike(self, tmp_path):
        """All example specs generate Spike extensions that compile."""
        from rvxv.generators.spike.spike_gen import SpikeGenerator

        for yaml_path in sorted(EXAMPLES_DIR.glob("*.yaml")):
            specs = load_spec(yaml_path)
            out = tmp_path / yaml_path.stem
            gen = SpikeGenerator()
            gen.generate(specs, out)

            spike_out = out / "spike"
            ext_cc = spike_out / "extension.cc"
            assert ext_cc.exists(), (
                f"{yaml_path.name}: extension.cc not generated"
            )
            _compile_spike_extension(ext_cc, spike_out)

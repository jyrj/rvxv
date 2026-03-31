"""Real end-to-end Spike test: generate extension, build, execute, verify.

This is the single most important test in RVXV. It proves that the generated
Spike C++ extension actually executes correctly — not just that it compiles.

Requires:
  - extern/spike submodule initialized and buildable
  - riscv64-linux-gnu-gcc cross-compiler
  - extern/riscv-tests for test framework headers
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rvxv.core.spec_parser import load_spec
from rvxv.generators.spike.spike_gen import SpikeGenerator
from rvxv.generators.tests.test_gen import AssemblyTestGenerator

REPO_ROOT = Path(__file__).parent.parent
SPIKE_SRC = REPO_ROOT / "extern" / "spike"
RISCV_TESTS = REPO_ROOT / "extern" / "riscv-tests"
SPIKE_BUILD = SPIKE_SRC / "build"
EXAMPLES_DIR = REPO_ROOT / "examples"


def _have_spike_build():
    return (SPIKE_BUILD / "spike").exists()


def _have_cross_compiler():
    return shutil.which("riscv64-linux-gnu-gcc") is not None


def _have_riscv_tests():
    return (RISCV_TESTS / "env" / "p" / "riscv_test.h").exists()


def _spike_needs_rebuild(ext_cc: Path) -> bool:
    """Check if the extension source is newer than libcustomext.so."""
    lib = SPIKE_BUILD / "libcustomext.so"
    if not lib.exists():
        return True
    return ext_cc.stat().st_mtime > lib.stat().st_mtime


def _rebuild_spike_extension(ext_cc: Path, encoding_h: Path, insn_dir: Path):
    """Copy generated files into Spike tree and rebuild."""
    shutil.copy2(encoding_h, SPIKE_SRC / "riscv" / "rvxv_encoding.h")
    shutil.copy2(ext_cc, SPIKE_SRC / "customext" / "rvxv_extension.cc")
    for insn_file in insn_dir.iterdir():
        if insn_file.suffix == ".h":
            shutil.copy2(insn_file, SPIKE_SRC / "riscv" / "insns" / insn_file.name)

    result = subprocess.run(
        ["make", f"-j{os.cpu_count() or 4}"],
        cwd=SPIKE_BUILD,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"Spike rebuild failed:\n{result.stderr[-2000:]}")


def _compile_test(s_file: Path, elf_file: Path):
    """Compile a .S test file into a bare-metal ELF."""
    result = subprocess.run(
        [
            "riscv64-linux-gnu-gcc",
            "-march=rv64gcv", "-mabi=lp64d",
            "-nostdlib", "-nostartfiles", "-static",
            f"-I{RISCV_TESTS / 'env' / 'p'}",
            f"-I{RISCV_TESTS / 'env'}",
            f"-I{RISCV_TESTS / 'isa' / 'macros' / 'scalar'}",
            f"-T{RISCV_TESTS / 'env' / 'p' / 'link.ld'}",
            str(s_file),
            "-o", str(elf_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Assembly compilation failed:\n{result.stderr}")


def _run_spike(elf_file: Path, log: bool = False) -> subprocess.CompletedProcess:
    """Run an ELF in Spike with the rvxv extension."""
    spike = str(SPIKE_BUILD / "spike")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(SPIKE_BUILD)

    cmd = [spike]
    if log:
        cmd.append("-l")
    cmd += ["--isa=rv64gcv", "--extension=rvxv", str(elf_file)]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)


skipif_no_spike = pytest.mark.skipif(
    not _have_spike_build(), reason="Spike not built (run: cd extern/spike/build && make)"
)
skipif_no_gcc = pytest.mark.skipif(
    not _have_cross_compiler(), reason="riscv64-linux-gnu-gcc not found"
)
skipif_no_tests = pytest.mark.skipif(
    not _have_riscv_tests(), reason="extern/riscv-tests not initialized"
)


@pytest.mark.e2e
@skipif_no_spike
@skipif_no_gcc
@skipif_no_tests
class TestSpikeRealE2E:
    """Generate extension, build into Spike, compile tests, EXECUTE them."""

    def _generate_and_run(self, spec_yaml: str, tmp_path: Path) -> subprocess.CompletedProcess:
        """Full pipeline: YAML -> C++ -> rebuild Spike -> compile .S -> run."""
        specs = load_spec(EXAMPLES_DIR / spec_yaml)

        # Generate Spike extension
        spike_gen = SpikeGenerator()
        gen_out = tmp_path / "gen"
        spike_gen.generate(specs, gen_out)

        # Generate tests
        test_gen = AssemblyTestGenerator()
        test_out = tmp_path / "tests"
        test_gen.generate(specs, gen_out)

        # Rebuild Spike with the new extension
        spike_out = gen_out / "spike"
        ext_cc = spike_out / "extension.cc"
        encoding_h = spike_out / "rvxv_encoding.h"
        insn_dir = spike_out / "insns"
        _rebuild_spike_extension(ext_cc, encoding_h, insn_dir)

        # Compile the directed test
        test_out = gen_out / "tests"
        directed = list((test_out / "directed").glob("*.S"))
        assert directed, "No directed test files generated"
        s_file = directed[0]
        elf_file = tmp_path / "test.elf"
        _compile_test(s_file, elf_file)

        # Run in Spike
        return _run_spike(elf_file)

    def test_int8_dot_product_e2e(self, tmp_path):
        """INT8 dot product: full E2E from YAML to Spike execution."""
        result = self._generate_and_run("int8_dot_product.yaml", tmp_path)
        assert result.returncode == 0, (
            f"Spike test FAILED (exit {result.returncode}).\n"
            f"This means the generated custom instruction produced wrong results.\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    def test_int8_dot_product_log_shows_custom_insn(self, tmp_path):
        """Verify that Spike actually executed the custom instruction."""
        specs = load_spec(EXAMPLES_DIR / "int8_dot_product.yaml")

        spike_gen = SpikeGenerator()
        gen_out = tmp_path / "gen"
        spike_gen.generate(specs, gen_out)

        test_gen = AssemblyTestGenerator()
        test_gen.generate(specs, gen_out)

        spike_out = gen_out / "spike"
        _rebuild_spike_extension(
            spike_out / "extension.cc",
            spike_out / "rvxv_encoding.h",
            spike_out / "insns",
        )

        s_file = list((gen_out / "tests" / "directed").glob("*.S"))[0]
        elf_file = tmp_path / "test.elf"
        _compile_test(s_file, elf_file)

        result = _run_spike(elf_file, log=True)
        assert result.returncode == 0

        # Check that the custom instruction was actually executed
        log = result.stderr
        assert "vdot4" in log.lower() or "0x8388045b" in log or "0x8188045b" in log, (
            "Spike log does not show any custom instruction execution. "
            "The test may have passed without actually running the custom instruction."
        )

        # Count executions — should match test structure
        custom_insn_count = log.count("8388045b") + log.count("8188045b")
        assert custom_insn_count >= 10, (
            f"Expected at least 10 custom instruction executions, got {custom_insn_count}"
        )

    def test_int32_reduction_sum_e2e(self, tmp_path):
        """INT32 reduction sum: verify reduction operation works E2E."""
        result = self._generate_and_run("int32_reduction_sum.yaml", tmp_path)
        assert result.returncode == 0, (
            f"Spike test FAILED for reduction_sum (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}"
        )

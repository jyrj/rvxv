"""End-to-end Verilator integration tests.

Generates RTL functional models from YAML specs, compiles with Verilator,
runs simulations, and checks results + SVA assertions.

Requires:
  - verilator on PATH
  - g++ with C++17 support
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rvxv.core.spec_parser import load_spec
from rvxv.generators.rtl.rtl_gen import RTLGenerator

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

HAS_VERILATOR = shutil.which("verilator") is not None
HAS_GPP = shutil.which("g++") is not None

skip_no_verilator = pytest.mark.skipif(
    not HAS_VERILATOR,
    reason="Verilator not found on PATH",
)


def _generate_rtl(spec_path: Path, output_dir: Path) -> Path:
    """Generate RTL artifacts and return the instruction directory."""
    specs = load_spec(spec_path)
    gen = RTLGenerator()
    gen.generate(specs, output_dir)

    # Return first instruction's RTL directory
    rtl_dir = output_dir / "rtl"
    insn_dirs = [d for d in rtl_dir.iterdir() if d.is_dir()]
    assert insn_dirs, f"No instruction directories in {rtl_dir}"
    return insn_dirs[0]


def _verilator_build(insn_dir: Path) -> subprocess.CompletedProcess:
    """Run 'make build' in the generated instruction directory."""
    return subprocess.run(
        ["make", "build"],
        cwd=insn_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _verilator_run(insn_dir: Path) -> subprocess.CompletedProcess:
    """Run 'make run' in the generated instruction directory."""
    return subprocess.run(
        ["make", "run"],
        cwd=insn_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )


# -----------------------------------------------------------------------
# Tests: RTL generation (no Verilator needed)
# -----------------------------------------------------------------------


class TestRTLGeneration:
    """Verify RTL artifacts are generated correctly (no tools needed)."""

    def test_int8_dot_generates_all_files(self, tmp_path):
        """INT8 dot product generates all RTL artifacts."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)

        assert (insn_dir / "vdot4_i8i8_acc32.sv").exists()
        assert (insn_dir / "tb_vdot4_i8i8_acc32.sv").exists()
        assert (insn_dir / "tb_vdot4_i8i8_acc32.cpp").exists()
        assert (insn_dir / "Makefile").exists()

    def test_int8_dot_rtl_content(self, tmp_path):
        """Generated RTL contains key structural elements."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)

        sv = (insn_dir / "vdot4_i8i8_acc32.sv").read_text()
        assert "module vdot4_i8i8_acc32" in sv
        assert "parameter VLEN" in sv
        assert "rvfi_valid" in sv
        assert "insn_match" in sv
        assert "GROUP_SZ" in sv

    def test_int8_dot_testbench_sva(self, tmp_path):
        """Testbench contains SVA properties."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)

        tb_sv = (insn_dir / "tb_vdot4_i8i8_acc32.sv").read_text()
        assert "property p_valid_decode" in tb_sv
        assert "property p_no_trap" in tb_sv
        assert "property p_rd_field" in tb_sv
        assert "property p_deterministic" in tb_sv
        assert "assert property" in tb_sv

    def test_int8_dot_cpp_vectors(self, tmp_path):
        """C++ testbench has embedded test vectors."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)

        cpp = (insn_dir / "tb_vdot4_i8i8_acc32.cpp").read_text()
        assert "TestVector" in cpp
        assert "get_test_vectors" in cpp
        assert "pack_elements" in cpp
        assert "RVFI trace" in cpp
        assert "PASS" in cpp
        assert "FAIL" in cpp

    def test_makefile_verilator_flags(self, tmp_path):
        """Makefile has correct Verilator flags."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)

        mk = (insn_dir / "Makefile").read_text()
        assert "--assert" in mk
        assert "--trace" in mk
        assert "--cc --exe --build" in mk
        assert "VLEN" in mk

    @pytest.mark.parametrize("spec_name", [
        "int8_dot_product",
        "bf16_fma",
        "fp8_e4m3_dot",
    ])
    def test_all_examples_generate_rtl(self, tmp_path, spec_name):
        """Each example spec generates valid RTL."""
        spec_path = EXAMPLES_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            pytest.skip(f"Example spec not found: {spec_name}")

        insn_dir = _generate_rtl(spec_path, tmp_path / spec_name)

        # Every instruction should produce 4 files
        sv_files = list(insn_dir.glob("*.sv"))
        cpp_files = list(insn_dir.glob("*.cpp"))
        assert len(sv_files) == 2, f"Expected 2 .sv files, got {len(sv_files)}"
        assert len(cpp_files) == 1, f"Expected 1 .cpp file, got {len(cpp_files)}"
        assert (insn_dir / "Makefile").exists()


# -----------------------------------------------------------------------
# Tests: Verilator compilation and simulation (requires Verilator)
# -----------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
class TestVerilatorE2E:
    """Full Verilator E2E: generate -> compile -> simulate -> check."""

    @skip_no_verilator
    def test_int8_dot_verilator_build(self, tmp_path):
        """INT8 dot product RTL compiles with Verilator."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)
        result = _verilator_build(insn_dir)
        assert result.returncode == 0, (
            f"Verilator build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Check the binary was produced
        obj_dir = insn_dir / "obj_dir"
        assert obj_dir.exists(), "obj_dir not created"

    @skip_no_verilator
    def test_int8_dot_verilator_run(self, tmp_path):
        """INT8 dot product simulation passes all test vectors."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)
        result = _verilator_run(insn_dir)
        assert result.returncode == 0, (
            f"Verilator simulation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Check output for PASS/FAIL summary
        assert "FAIL" not in result.stdout or "0 FAIL" in result.stdout, (
            f"Test failures detected:\n{result.stdout}"
        )

    @skip_no_verilator
    def test_int8_dot_rvfi_trace(self, tmp_path):
        """Simulation produces RVFI trace CSV."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)
        result = _verilator_run(insn_dir)
        assert result.returncode == 0, f"Simulation failed:\n{result.stderr}"

        rvfi_csv = insn_dir / "rvfi_trace_vdot4_i8i8_acc32.csv"
        assert rvfi_csv.exists(), "RVFI trace CSV not produced"

        content = rvfi_csv.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 2, "RVFI trace has no data rows"
        assert "rvfi_valid" in lines[0], "RVFI trace missing header"

    @skip_no_verilator
    def test_int8_dot_sva_no_violations(self, tmp_path):
        """No SVA assertion violations during simulation."""
        insn_dir = _generate_rtl(EXAMPLES_DIR / "int8_dot_product.yaml", tmp_path)
        result = _verilator_run(insn_dir)
        assert result.returncode == 0, f"Simulation failed:\n{result.stderr}"

        # Verilator SVA violations appear in stderr as "Assertion failed"
        assert "Assertion failed" not in result.stderr, (
            f"SVA assertion violation:\n{result.stderr}"
        )
        assert "[RVXV-SVA]" not in result.stderr, (
            f"RVXV SVA violation:\n{result.stderr}"
        )

    @skip_no_verilator
    @pytest.mark.parametrize("spec_name", [
        "int8_dot_product",
        pytest.param("bf16_fma", marks=pytest.mark.xfail(
            reason="RTL template uses integer math; BF16 needs FP functional unit")),
        pytest.param("fp8_e4m3_dot", marks=pytest.mark.xfail(
            reason="RTL template uses integer math; FP8 needs FP functional unit")),
    ])
    def test_example_specs_verilator(self, tmp_path, spec_name):
        """Each example spec builds and simulates successfully."""
        spec_path = EXAMPLES_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            pytest.skip(f"Example spec not found: {spec_name}")

        insn_dir = _generate_rtl(spec_path, tmp_path / spec_name)

        build_result = _verilator_build(insn_dir)
        assert build_result.returncode == 0, (
            f"Verilator build failed for {spec_name}:\n{build_result.stderr}"
        )

        run_result = _verilator_run(insn_dir)
        assert run_result.returncode == 0, (
            f"Verilator sim failed for {spec_name}:\n"
            f"stdout: {run_result.stdout}\nstderr: {run_result.stderr}"
        )

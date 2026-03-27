"""End-to-end Spike integration tests.

Generates RVXV Spike extension code from YAML specs, patches Spike source,
rebuilds Spike, compiles assembly tests, and runs them through Spike.

Requires:
  - Spike submodule at extern/spike (with prior build)
  - riscv64-linux-gnu-gcc on PATH
  - riscv-tests submodule at extern/riscv-tests
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SPIKE_SRC = REPO_ROOT / "extern" / "spike"
RISCV_TESTS = REPO_ROOT / "extern" / "riscv-tests"
BUILD_DIR = REPO_ROOT / "build" / "e2e"
SPIKE_BUILD = BUILD_DIR / "spike-build"
SPIKE_INSTALL = BUILD_DIR / "spike-install"
SPIKE_BIN = SPIKE_INSTALL / "bin" / "spike"

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None
HAS_SPIKE_SUBMODULE = (SPIKE_SRC / "riscv" / "extension.h").is_file()
HAS_RISCV_TESTS = (RISCV_TESTS / "env" / "p" / "riscv_test.h").is_file()


def _spike_available() -> bool:
    """Check if Spike binary is built and ready."""
    return SPIKE_BIN.is_file() and os.access(SPIKE_BIN, os.X_OK)


skip_no_spike_e2e = pytest.mark.skipif(
    not (HAS_RISCV_GCC and HAS_SPIKE_SUBMODULE and HAS_RISCV_TESTS),
    reason="Spike E2E requires riscv-gcc, Spike submodule, and riscv-tests",
)


@pytest.fixture(scope="session")
def spike_binary() -> Path:
    """Ensure Spike is built and return the binary path.

    Uses the existing build if available. Rebuilds only if the binary
    is missing.
    """
    if _spike_available():
        return SPIKE_BIN

    if not HAS_SPIKE_SUBMODULE:
        pytest.skip("Spike submodule not available")

    SPIKE_BUILD.mkdir(parents=True, exist_ok=True)
    SPIKE_INSTALL.mkdir(parents=True, exist_ok=True)

    nprocs = os.cpu_count() or 4

    # Configure
    if not (SPIKE_BUILD / "Makefile").exists():
        subprocess.run(
            [str(SPIKE_SRC / "configure"), f"--prefix={SPIKE_INSTALL}"],
            cwd=SPIKE_BUILD,
            check=True,
            capture_output=True,
        )

    # Build and install
    subprocess.run(
        ["make", f"-j{nprocs}"],
        cwd=SPIKE_BUILD,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["make", "install"],
        cwd=SPIKE_BUILD,
        check=True,
        capture_output=True,
    )

    assert SPIKE_BIN.is_file(), f"Spike build failed — {SPIKE_BIN} not found"
    return SPIKE_BIN


def _generate_and_patch_spike(spec_path: Path, output_dir: Path) -> None:
    """Generate RVXV artifacts and patch Spike source tree."""
    subprocess.run(
        [
            "python3", "-m", "rvxv.cli", "generate",
            "--spec", str(spec_path),
            "--output", str(output_dir),
            "--targets", "spike,tests",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    spike_out = output_dir / "spike"
    assert (spike_out / "extension.cc").exists()

    # Copy into Spike source tree
    shutil.copy2(spike_out / "rvxv_encoding.h", SPIKE_SRC / "riscv")
    shutil.copy2(spike_out / "extension.cc", SPIKE_SRC / "customext" / "rvxv_extension.cc")
    shutil.copy2(spike_out / "disasm.cc", SPIKE_SRC / "customext" / "rvxv_disasm.cc")

    for insn_h in (spike_out / "insns").glob("*.h"):
        shutil.copy2(insn_h, SPIKE_SRC / "riscv" / "insns")

    # Ensure customext.mk.in includes RVXV sources
    customext_mk = SPIKE_SRC / "customext" / "customext.mk.in"
    content = customext_mk.read_text()
    if "rvxv_extension.cc" not in content:
        content = content.replace(
            "customext_srcs = \\",
            "customext_srcs = \\\n\trvxv_extension.cc \\\n\trvxv_disasm.cc \\",
        )
        customext_mk.write_text(content)


def _rebuild_spike() -> None:
    """Rebuild Spike with the patched extension."""
    nprocs = os.cpu_count() or 4
    subprocess.run(
        ["make", f"-j{nprocs}"],
        cwd=SPIKE_BUILD,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["make", "install"],
        cwd=SPIKE_BUILD,
        check=True,
        capture_output=True,
    )


def _compile_asm_tests(output_dir: Path, bin_dir: Path) -> list[Path]:
    """Compile generated .S files into ELF binaries."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    asm_files = list((output_dir / "tests").rglob("*.S"))
    assert asm_files, f"No .S files found in {output_dir / 'tests'}"

    link_script = RISCV_TESTS / "env" / "p" / "link.ld"
    include_dirs = [
        f"-I{RISCV_TESTS / 'env' / 'p'}",
        f"-I{RISCV_TESTS / 'env'}",
        f"-I{RISCV_TESTS / 'isa' / 'macros' / 'scalar'}",
    ]

    elfs: list[Path] = []
    for asm in asm_files:
        elf = bin_dir / f"{asm.stem}.elf"
        result = subprocess.run(
            [
                "riscv64-linux-gnu-gcc",
                "-march=rv64gcv", "-mabi=lp64d",
                *include_dirs,
                "-nostdlib", "-nostartfiles", "-static",
                "-T", str(link_script),
                str(asm), "-o", str(elf),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"Assembly failed for {asm.name}:\n{result.stderr}"
        )
        elfs.append(elf)

    return elfs


def _run_spike(spike_bin: Path, elf: Path) -> subprocess.CompletedProcess:
    """Run a single ELF through Spike with timeout."""
    extlib = None
    for candidate in [
        SPIKE_BUILD / "customext" / "libcustomext.so",
        SPIKE_INSTALL / "lib" / "libcustomext.so",
    ]:
        if candidate.is_file():
            extlib = candidate
            break

    cmd = [str(spike_bin), "--isa=rv64gcv"]
    if extlib:
        cmd += [f"--extlib={extlib}", "--extension=rvxv"]
    cmd.append(str(elf))

    return subprocess.run(cmd, capture_output=True, text=True, timeout=10)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
class TestSpikeE2E:
    """End-to-end Spike tests: generate -> patch -> build -> run."""

    @skip_no_spike_e2e
    def test_int8_dot_product_spike(self, spike_binary, tmp_path):
        """INT8 dot product passes all directed and random tests in Spike."""
        spec_path = EXAMPLES_DIR / "int8_dot_product.yaml"
        rvxv_out = tmp_path / "rvxv-gen"
        test_bin = tmp_path / "test-bin"

        _generate_and_patch_spike(spec_path, rvxv_out)
        _rebuild_spike()
        elfs = _compile_asm_tests(rvxv_out, test_bin)

        for elf in elfs:
            result = _run_spike(spike_binary, elf)
            assert result.returncode == 0, (
                f"Spike FAIL for {elf.name} (exit={result.returncode}):\n"
                f"{result.stderr}"
            )

    @skip_no_spike_e2e
    @pytest.mark.parametrize("spec_name", [
        "int8_dot_product",
        pytest.param("bf16_fma", marks=pytest.mark.xfail(
            reason="BF16 random test golden values have rounding edge cases")),
        pytest.param("fp8_e4m3_dot", marks=pytest.mark.xfail(
            reason="FP8 random test golden values have precision edge cases")),
    ])
    def test_example_specs_spike(self, spike_binary, tmp_path, spec_name):
        """Each example spec generates tests that pass in Spike."""
        spec_path = EXAMPLES_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            pytest.skip(f"Example spec not found: {spec_name}")

        rvxv_out = tmp_path / "rvxv-gen"
        test_bin = tmp_path / "test-bin"

        _generate_and_patch_spike(spec_path, rvxv_out)
        _rebuild_spike()
        elfs = _compile_asm_tests(rvxv_out, test_bin)

        for elf in elfs:
            result = _run_spike(spike_binary, elf)
            assert result.returncode == 0, (
                f"Spike FAIL for {spec_name}/{elf.name} (exit={result.returncode}):\n"
                f"{result.stderr}"
            )


@pytest.mark.e2e
@pytest.mark.slow
class TestSpikeArtifacts:
    """Verify Spike artifacts are structurally correct."""

    @skip_no_spike_e2e
    def test_generated_extension_structure(self, tmp_path):
        """Generated Spike extension has all required files."""
        spec_path = EXAMPLES_DIR / "int8_dot_product.yaml"
        subprocess.run(
            [
                "python3", "-m", "rvxv.cli", "generate",
                "--spec", str(spec_path),
                "--output", str(tmp_path),
                "--targets", "spike,tests",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )

        spike_dir = tmp_path / "spike"
        assert (spike_dir / "extension.cc").exists()
        assert (spike_dir / "disasm.cc").exists()
        assert (spike_dir / "rvxv_encoding.h").exists()
        assert list((spike_dir / "insns").glob("*.h"))

        tests_dir = tmp_path / "tests"
        assert list(tests_dir.rglob("*.S"))

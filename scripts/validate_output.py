#!/usr/bin/env python3
"""Validate RVXV generated output directory.

Checks that all expected files exist and contain required patterns.
Optionally compiles assembly files with riscv64-linux-gnu-gcc.

Usage:
    python scripts/validate_output.py /path/to/generated/output
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


class Check:
    """A single validation check with a name and pass/fail status."""

    def __init__(self, name):
        self.name = name
        self.passed = False
        self.detail = ""

    def pass_(self, detail=""):
        self.passed = True
        self.detail = detail

    def fail(self, detail=""):
        self.passed = False
        self.detail = detail


def find_files(directory, extension):
    """Recursively find files with a given extension."""
    results = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                results.append(os.path.join(root, f))
    return sorted(results)


def check_directory_structure(output_dir):
    """Check that all expected subdirectories exist."""
    check = Check("Directory structure (spike/, tests/, assertions/, docs/)")
    expected = ["spike", "tests", "assertions", "docs"]
    missing = []
    for d in expected:
        path = os.path.join(output_dir, d)
        if not os.path.isdir(path):
            missing.append(d)
    if missing:
        check.fail("Missing directories: " + ", ".join(missing))
    else:
        check.pass_("All expected directories present")
    return check


def check_encoding_header(output_dir):
    """Check that rvxv_encoding.h has MATCH/MASK defines."""
    check = Check("Encoding header has MATCH/MASK defines")
    header_files = find_files(output_dir, "rvxv_encoding.h")
    if not header_files:
        # Also check for encoding.h as a fallback
        header_files = find_files(output_dir, "encoding.h")
    if not header_files:
        check.fail("No rvxv_encoding.h (or encoding.h) found")
        return check

    header = header_files[0]
    with open(header) as f:
        content = f.read()

    has_match = bool(re.search(r"#define\s+MATCH_", content))
    has_mask = bool(re.search(r"#define\s+MASK_", content))

    if has_match and has_mask:
        match_count = len(re.findall(r"#define\s+MATCH_", content))
        mask_count = len(re.findall(r"#define\s+MASK_", content))
        check.pass_(
            f"{header}: {match_count} MATCH, {mask_count} MASK defines"
        )
    else:
        problems = []
        if not has_match:
            problems.append("no MATCH_ defines")
        if not has_mask:
            problems.append("no MASK_ defines")
        check.fail(f"{header}: " + ", ".join(problems))
    return check


def check_vstart_handling(output_dir):
    """Check that execute body has vstart handling."""
    check = Check("Execute body has vstart handling")
    # Look in spike directory for C/C++ files
    spike_dir = os.path.join(output_dir, "spike")
    if not os.path.isdir(spike_dir):
        check.fail("spike/ directory not found")
        return check

    cpp_files = find_files(spike_dir, ".h") + find_files(spike_dir, ".cc")
    if not cpp_files:
        check.fail("No .h or .cc files found in spike/")
        return check

    found_vstart = False
    found_in = None
    for path in cpp_files:
        with open(path) as f:
            content = f.read()
        if re.search(r"vstart|VI_CHECK_S[SD]S|P\.VU\.vstart", content, re.IGNORECASE):
            found_vstart = True
            found_in = path
            break

    if found_vstart:
        check.pass_(f"vstart reference found in {found_in}")
    else:
        check.fail("No vstart handling found in spike/ sources")
    return check


def check_dot_word_encoding(output_dir):
    """Check that assembly tests use .word (not .insn r)."""
    check = Check("Assembly tests use .word (not .insn r)")
    tests_dir = os.path.join(output_dir, "tests")
    if not os.path.isdir(tests_dir):
        check.fail("tests/ directory not found")
        return check

    asm_files = find_files(tests_dir, ".S") + find_files(tests_dir, ".s")
    if not asm_files:
        check.fail("No assembly files found in tests/")
        return check

    has_dot_word = False
    has_insn_r = False
    for path in asm_files:
        with open(path) as f:
            content = f.read()
        if re.search(r"\.word\s+0x", content):
            has_dot_word = True
        if re.search(r"\.insn\s+r", content):
            has_insn_r = True

    if has_dot_word and not has_insn_r:
        check.pass_(f"Uses .word encoding ({len(asm_files)} assembly file(s))")
    elif has_insn_r:
        check.fail("Found .insn r usage -- should use .word instead")
    else:
        check.fail("No .word 0x... encoding found in assembly files")
    return check


def check_golden_values_and_bne(output_dir):
    """Check that assembly has golden values and bne comparisons."""
    check = Check("Assembly has golden values and bne comparisons")
    tests_dir = os.path.join(output_dir, "tests")
    if not os.path.isdir(tests_dir):
        check.fail("tests/ directory not found")
        return check

    asm_files = find_files(tests_dir, ".S") + find_files(tests_dir, ".s")
    if not asm_files:
        check.fail("No assembly files found in tests/")
        return check

    has_golden = False
    has_bne = False
    for path in asm_files:
        with open(path) as f:
            content = f.read()
        # Golden values: look for expected/golden labels or known constant loads
        if re.search(r"(golden|expected|li\s+\w+,\s*0x|\.word\s+0x)", content, re.IGNORECASE):
            has_golden = True
        if re.search(r"\bbne\b", content):
            has_bne = True

    if has_golden and has_bne:
        check.pass_("Golden values and bne comparisons found")
    else:
        problems = []
        if not has_golden:
            problems.append("no golden/expected values detected")
        if not has_bne:
            problems.append("no bne comparisons found")
        check.fail("; ".join(problems))
    return check


def check_gcc_compile(output_dir):
    """Optionally compile .S files with riscv64-linux-gnu-gcc."""
    check = Check("Assembly compiles with riscv64-linux-gnu-gcc")
    gcc = shutil.which("riscv64-linux-gnu-gcc")
    if gcc is None:
        check.pass_("SKIP -- riscv64-linux-gnu-gcc not found on PATH")
        return check

    tests_dir = os.path.join(output_dir, "tests")
    asm_files = find_files(tests_dir, ".S") if os.path.isdir(tests_dir) else []
    if not asm_files:
        check.pass_("SKIP -- no .S files to compile")
        return check

    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal riscv_test.h and test_macros.h stubs
        stub_h = os.path.join(tmpdir, "riscv_test.h")
        with open(stub_h, "w") as f:
            f.write(
                "#define RVTEST_RV64UV\n"
                "#define RVTEST_CODE_BEGIN "
                ".text; .globl _start; _start:\n"
                "#define RVTEST_CODE_END\n"
                "#define RVTEST_DATA_BEGIN\n"
                "#define RVTEST_DATA_END\n"
                "#define TEST_PASSFAIL j _start\n"
                "#define TESTNUM t6\n"
            )
        with open(os.path.join(tmpdir, "test_macros.h"), "w") as f:
            f.write("/* stub */\n")

        for path in asm_files:
            # Rewrite #include to use our stubs
            with open(path) as f:
                src = f.read()
            src = src.replace(
                '#include "riscv_test.h"',
                f'#include "{stub_h}"',
            )
            src = src.replace(
                '#include "test_macros.h"',
                f'#include "{os.path.join(tmpdir, "test_macros.h")}"',
            )
            patched = os.path.join(
                tmpdir, os.path.basename(path),
            )
            with open(patched, "w") as f:
                f.write(src)

            obj = patched + ".o"
            result = subprocess.run(
                [gcc, "-march=rv64gcv", "-c", patched, "-o", obj],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                failures.append(
                    f"{os.path.basename(path)}: "
                    f"{result.stderr.strip()}"
                )

    if not failures:
        check.pass_(f"All {len(asm_files)} file(s) compiled successfully")
    else:
        check.fail("\n    ".join(failures))
    return check


def main():
    parser = argparse.ArgumentParser(
        description="Validate RVXV generated output directory.",
        epilog="Exit code 0 if all checks pass, 1 if any fail.",
    )
    parser.add_argument(
        "output_dir",
        help="Path to RVXV generated output directory",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: {output_dir} is not a directory")
        sys.exit(1)

    print(f"Validating RVXV output: {output_dir}\n")

    checks = [
        check_directory_structure(output_dir),
        check_encoding_header(output_dir),
        check_vstart_handling(output_dir),
        check_dot_word_encoding(output_dir),
        check_golden_values_and_bne(output_dir),
        check_gcc_compile(output_dir),
    ]

    # Print results
    all_passed = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        if not c.passed:
            all_passed = False
        print(f"  [{status}] {c.name}")
        if c.detail:
            print(f"         {c.detail}")

    # Summary
    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    print()
    if all_passed:
        print(f"RESULT: ALL CHECKS PASSED ({passed}/{total})")
    else:
        failed = total - passed
        print(f"RESULT: {failed} CHECK(S) FAILED ({passed}/{total} passed)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

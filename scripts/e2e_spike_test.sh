#!/usr/bin/env bash
# End-to-end Spike validation for RVXV
#
# Builds Spike from source, generates RVXV extension code,
# patches Spike's customext with the RVXV extension, rebuilds,
# compiles assembly tests, and runs them through Spike.
#
# Prerequisites:
#   sudo dnf install boost-devel automake libtool  # Fedora
#   # or: sudo apt install libboost-dev automake libtool  # Ubuntu
#   riscv64-linux-gnu-gcc must be on PATH
#
# Usage:
#   ./scripts/e2e_spike_test.sh [--spec path/to/spec.yaml] [--clean]
#
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPIKE_SRC="$PROJ_ROOT/extern/spike"
BUILD_DIR="$PROJ_ROOT/build/e2e"
SPIKE_BUILD="$BUILD_DIR/spike-build"
SPIKE_INSTALL="$BUILD_DIR/spike-install"
RVXV_OUT="$BUILD_DIR/rvxv-gen"
TEST_BIN_DIR="$BUILD_DIR/test-bin"
RISCV_TESTS="$PROJ_ROOT/extern/riscv-tests"

# Default spec
SPEC="${SPEC:-$PROJ_ROOT/examples/int8_dot_product.yaml}"
CLEAN=false

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec)   SPEC="$2"; shift 2 ;;
    --clean)  CLEAN=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--spec path/to/spec.yaml] [--clean]"
      echo "  --spec   YAML instruction spec (default: examples/int8_dot_product.yaml)"
      echo "  --clean  Remove build directory and start fresh"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# -----------------------------------------------------------------------
# Step 0: Preflight checks
# -----------------------------------------------------------------------
info "Preflight checks..."

command -v riscv64-linux-gnu-gcc >/dev/null 2>&1 || \
  fail "riscv64-linux-gnu-gcc not found. Install: dnf install gcc-riscv64-linux-gnu"
command -v g++ >/dev/null 2>&1 || \
  fail "g++ not found. Install a C++ compiler."
command -v python3 >/dev/null 2>&1 || \
  fail "python3 not found."

# Check boost headers
if ! echo '#include <boost/algorithm/string.hpp>' | g++ -x c++ -fsyntax-only - 2>/dev/null; then
  fail "boost-devel headers not found. Install: dnf install boost-devel"
fi

if $CLEAN && [ -d "$BUILD_DIR" ]; then
  info "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" "$SPIKE_BUILD" "$SPIKE_INSTALL" "$RVXV_OUT" "$TEST_BIN_DIR"

NPROCS="$(nproc 2>/dev/null || echo 4)"

# -----------------------------------------------------------------------
# Step 1: Build Spike (if not already built)
# -----------------------------------------------------------------------
if [ ! -x "$SPIKE_INSTALL/bin/spike" ]; then
  info "Building Spike from source ($NPROCS threads)..."

  cd "$SPIKE_BUILD"

  if [ ! -f Makefile ]; then
    "$SPIKE_SRC/configure" \
      --prefix="$SPIKE_INSTALL" \
      2>&1 | tail -5
  fi

  make -j"$NPROCS" 2>&1 | tail -5
  make install 2>&1 | tail -3

  if [ ! -x "$SPIKE_INSTALL/bin/spike" ]; then
    fail "Spike build failed — no spike binary produced."
  fi
  info "Spike built successfully: $SPIKE_INSTALL/bin/spike"
else
  info "Spike already built: $SPIKE_INSTALL/bin/spike"
fi

SPIKE_BIN="$SPIKE_INSTALL/bin/spike"

# -----------------------------------------------------------------------
# Step 2: Generate RVXV artifacts
# -----------------------------------------------------------------------
info "Generating RVXV artifacts from: $SPEC"

cd "$PROJ_ROOT"
python3 -m rvxv.cli generate --spec "$SPEC" --output "$RVXV_OUT" --targets spike,tests

if [ ! -f "$RVXV_OUT/spike/extension.cc" ]; then
  fail "RVXV generation failed — no extension.cc produced."
fi

INSN_COUNT=$(ls "$RVXV_OUT/spike/insns/"*.h 2>/dev/null | wc -l)
info "Generated: extension.cc, disasm.cc, rvxv_encoding.h, $INSN_COUNT instruction body(ies)"

# -----------------------------------------------------------------------
# Step 3: Patch Spike customext with RVXV extension and rebuild
# -----------------------------------------------------------------------
info "Patching Spike source tree with RVXV extension..."

# Copy generated files into Spike source tree
cp "$RVXV_OUT/spike/rvxv_encoding.h" "$SPIKE_SRC/riscv/"
cp "$RVXV_OUT/spike/extension.cc"    "$SPIKE_SRC/customext/rvxv_extension.cc"
cp "$RVXV_OUT/spike/disasm.cc"       "$SPIKE_SRC/customext/rvxv_disasm.cc"

# Copy instruction execute bodies
for f in "$RVXV_OUT/spike/insns/"*.h; do
  cp "$f" "$SPIKE_SRC/riscv/insns/"
done

# Update customext.mk.in to include RVXV sources
CUSTOMEXT_MK="$SPIKE_SRC/customext/customext.mk.in"
if ! grep -q "rvxv_extension.cc" "$CUSTOMEXT_MK"; then
  info "Adding RVXV sources to customext.mk.in"
  sed -i 's|^customext_srcs = \\|customext_srcs = \\\n\trvxv_extension.cc \\\n\trvxv_disasm.cc \\|' "$CUSTOMEXT_MK"
fi

# Rebuild Spike with the extension
info "Rebuilding Spike with RVXV extension ($NPROCS threads)..."
cd "$SPIKE_BUILD"
make -j"$NPROCS" 2>&1 | tail -5
make install 2>&1 | tail -3

# Verify the shared library was built
EXTLIB=$(find "$SPIKE_BUILD" "$SPIKE_INSTALL" -name "libcustomext.so" 2>/dev/null | head -1)
if [ -z "$EXTLIB" ]; then
  # Try static linking — extension might be built into spike directly
  warn "No libcustomext.so found; extension may be statically linked."
  EXTLIB=""
fi

info "Spike rebuilt with RVXV extension."

# -----------------------------------------------------------------------
# Step 4: Compile assembly tests
# -----------------------------------------------------------------------
info "Compiling generated assembly tests..."

ASM_FILES=$(find "$RVXV_OUT/tests" -name "*.S" 2>/dev/null)
if [ -z "$ASM_FILES" ]; then
  fail "No .S assembly files found in $RVXV_OUT/tests/"
fi

COMPILE_OK=0
COMPILE_FAIL=0
GCC=riscv64-linux-gnu-gcc

# Include paths for riscv-tests framework
INCLUDE_DIRS="-I$RISCV_TESTS/env/p -I$RISCV_TESTS/env -I$RISCV_TESTS/isa/macros/scalar"
LINK_SCRIPT="$RISCV_TESTS/env/p/link.ld"

for asm_file in $ASM_FILES; do
  base=$(basename "$asm_file" .S)
  obj_file="$TEST_BIN_DIR/${base}.o"
  elf_file="$TEST_BIN_DIR/${base}.elf"

  # Compile
  if $GCC -march=rv64gcv -mabi=lp64d $INCLUDE_DIRS \
      -nostdlib -nostartfiles -static \
      -T "$LINK_SCRIPT" \
      "$asm_file" -o "$elf_file" 2>"$TEST_BIN_DIR/${base}.compile.log"; then
    COMPILE_OK=$((COMPILE_OK + 1))
  else
    COMPILE_FAIL=$((COMPILE_FAIL + 1))
    warn "Failed to compile: $base"
    cat "$TEST_BIN_DIR/${base}.compile.log"
  fi
done

info "Compilation: $COMPILE_OK passed, $COMPILE_FAIL failed"

if [ "$COMPILE_OK" -eq 0 ]; then
  fail "No tests compiled successfully."
fi

# -----------------------------------------------------------------------
# Step 5: Run tests through Spike
# -----------------------------------------------------------------------
info "Running tests through Spike..."

RUN_PASS=0
RUN_FAIL=0
RUN_ERRORS=()

for elf_file in "$TEST_BIN_DIR"/*.elf; do
  base=$(basename "$elf_file" .elf)

  # Build spike command
  SPIKE_CMD="$SPIKE_BIN --isa=rv64gcv"
  if [ -n "${EXTLIB:-}" ]; then
    SPIKE_CMD="$SPIKE_CMD --extlib=$EXTLIB --extension=rvxv"
  fi
  SPIKE_CMD="$SPIKE_CMD $elf_file"

  # Run with timeout (10 seconds)
  if timeout 10 $SPIKE_CMD >"$TEST_BIN_DIR/${base}.spike.log" 2>&1; then
    RUN_PASS=$((RUN_PASS + 1))
    info "  PASS: $base"
  else
    EXIT_CODE=$?
    RUN_FAIL=$((RUN_FAIL + 1))
    # Decode which test failed from exit code
    if [ $EXIT_CODE -eq 124 ]; then
      RUN_ERRORS+=("$base: TIMEOUT")
      warn "  TIMEOUT: $base"
    else
      # Exit code encodes TESTNUM: (TESTNUM << 1) | 1
      if [ $((EXIT_CODE & 1)) -eq 1 ]; then
        FAILED_TEST=$((EXIT_CODE >> 1))
        RUN_ERRORS+=("$base: test case $FAILED_TEST failed (exit=$EXIT_CODE)")
        warn "  FAIL: $base — test case $FAILED_TEST (exit code $EXIT_CODE)"
      else
        RUN_ERRORS+=("$base: exit code $EXIT_CODE")
        warn "  FAIL: $base — exit code $EXIT_CODE"
      fi
    fi
    # Show last few lines of spike log
    tail -5 "$TEST_BIN_DIR/${base}.spike.log" 2>/dev/null || true
  fi
done

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  RVXV End-to-End Spike Validation Summary"
echo "=============================================="
echo "  Spec:           $(basename "$SPEC")"
echo "  Instructions:   $INSN_COUNT"
echo "  Tests compiled: $COMPILE_OK / $((COMPILE_OK + COMPILE_FAIL))"
echo "  Tests passed:   $RUN_PASS / $((RUN_PASS + RUN_FAIL))"

if [ ${#RUN_ERRORS[@]} -gt 0 ]; then
  echo ""
  echo "  Failures:"
  for err in "${RUN_ERRORS[@]}"; do
    echo "    - $err"
  done
fi

echo "=============================================="
echo "  Build dir: $BUILD_DIR"
echo "  Spike:     $SPIKE_BIN"
echo "  Tests:     $TEST_BIN_DIR/"
echo "=============================================="

if [ "$RUN_FAIL" -gt 0 ]; then
  exit 1
fi
info "All tests passed!"

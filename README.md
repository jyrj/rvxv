# RVXV -- RISC-V AI Extension Verification Platform

RVXV generates verification infrastructure for custom RISC-V AI vector
instructions from YAML specifications. Given one YAML file describing an
instruction's encoding, operands, and semantics, it produces Spike C++
extensions, self-checking RISC-V assembly tests, SVA assertions, and
documentation.

## What Problem This Solves

RISC-V AI chip companies (SiFive, Tenstorrent, Esperanto, etc.) each add
custom vector instructions for operations like INT8 dot products, BF16 FMA,
and quantized matrix multiply. Verifying each instruction requires:

- A Spike golden reference model (C++ extension)
- Directed and random assembly tests with expected values
- Formal properties (SVA) for decode and functional correctness
- RVFI trace comparison infrastructure

Most of this is repetitive infrastructure -- the same decode patterns, the same
vector loop structure, the same mask handling -- just with different opcodes and
element types. RVXV automates the boilerplate.

## What Works Today (v0.1.0)

Tested and verified with real tools:

- **Spike C++ extensions** compile against real Spike headers (`extern/spike`)
  using `g++ -fsyntax-only`. The generated code uses the correct `insn_desc_t`
  (10 fields), `extension_t` method signatures, and softfloat `float32_t`
  conversions.
- **Assembly tests** assemble with `riscv64-linux-gnu-gcc -march=rv64gcv`. Uses
  `.word` encoding (not `.insn r`) because GCC does not support `.insn r` with
  vector register operands and large funct7 values.
- **RVV encoding** is correct: `format: vector` with `funct6` properly separates
  the vm bit (bit 25) from the instruction encoding. MASK does not include bit 25,
  so both masked and unmasked variants share the same MATCH/MASK pair.
- **End-to-end Spike execution** verified: the generated INT8 dot product
  extension was built into Spike, test binaries were compiled and executed,
  and all golden values matched. This proves the generated C++ executes
  correctly in the reference simulator.
- **Golden values** are independently verified against manual Python computation
  (not computed by the same engine that generates them).
- **Numeric library** is cross-validated against independent IEEE 754 references
  (`struct.pack`, manual bit-pattern computation) for FP8 E4M3, FP8 E5M2,
  BFloat16, INT4, and MX formats.

207 tests (202 pass, 5 expected failures), including real toolchain compilation
tests and end-to-end Spike execution tests.

## What Has NOT Been Tested

Be honest about what we haven't proven:

- **Limited end-to-end Spike execution.** INT8 dot product has been fully tested
  end-to-end (generated extension built into Spike, tests compiled and run with
  correct results). Other operation types (FMA, reduction, conversion) have not
  been tested end-to-end yet. The reduction sum test revealed a golden-value bug
  when run in Spike — this is exactly what E2E tests are for.
- **SVA assertions are structural templates.** They produce syntactically valid
  SystemVerilog but have not been tested in a real simulation environment
  (VCS, Questa, etc.).
- **RVFI checker is a template.** The Python trace checker script would need
  adaptation for your specific trace format.
- **Only tested with GCC 15.** Other cross-compiler versions may behave
  differently.

## Quick Start

```bash
pip install -e .

# Generate from a YAML spec
rvxv generate --spec examples/int8_dot_product.yaml --output ./build/

# Validate a spec without generating
rvxv validate --spec examples/int8_dot_product.yaml

# Use a preset
rvxv preset --list
rvxv preset --name common_ai_ops --output ./build/common/

# Export JSON Schema for editor autocompletion
rvxv schema --output ./instruction_spec.schema.json
```

## Real Example: INT8 Dot Product

Here is the complete workflow from YAML spec to generated artifacts.

### 1. The YAML Spec

```yaml
name: vdot4_i8i8_acc32
description: "4-way INT8 dot product with INT32 accumulation"
encoding:
  format: vector
  opcode: 0x5b    # custom-2
  funct3: 0x0
  funct6: 0x20    # bits[31:26], vm bit[25] is separate
operands:
  vs2: { type: vector, element: int8, groups: 4 }
  vs1: { type: vector, element: int8, groups: 4 }
  vd:  { type: vector, element: int32 }
semantics:
  operation: dot_product
  accumulate: true
  saturation: none
constraints:
  vl_dependent: true
  mask_agnostic: true
  tail_agnostic: true
```

### 2. Generated Files (13 total)

```
build/
├── spike/
│   ├── rvxv_encoding.h                        # MATCH/MASK defines
│   ├── insns/vdot4_i8i8_acc32.h               # Execute function body
│   ├── extension.cc                           # Extension registration
│   ├── disasm.cc                              # Disassembly support
│   └── Makefile.patch                         # Build integration
├── tests/
│   ├── directed/vdot4_i8i8_acc32_directed.S   # Corner case tests
│   └── random/vdot4_i8i8_acc32_random.S       # Random vector tests
├── assertions/
│   ├── vdot4_i8i8_acc32_assertions.sv         # SVA decode + result
│   ├── rvxv_coverage.sv                       # Functional coverage
│   ├── rvxv_bind.sv                           # Bind file
│   └── rvfi_checker.py                        # RVFI trace checker
└── docs/
    ├── vdot4_i8i8_acc32.md
    └── instruction_summary.md
```

### 3. What the Generated Code Looks Like

**Encoding header** (`rvxv_encoding.h`):
```c
#define MATCH_VDOT4_I8I8_ACC32  0x8000005B
#define MASK_VDOT4_I8I8_ACC32   0xFC00707F
```

**Execute body** (`insns/vdot4_i8i8_acc32.h`) -- this is the Spike instruction
implementation:
```c
require_vector_vs;
require(P.VU.vsew == e8);
{
  const reg_t vl = P.VU.vl->read();
  const reg_t vstart_val = P.VU.vstart->read();
  const bool masked = insn.v_vm() == 0;
  const reg_t rd_num = insn.rd();
  const reg_t rs2_num = insn.rs2();
  const reg_t rs1_num = insn.rs1();

  const reg_t num_outputs = vl / 4;
  for (reg_t i = vstart_val; i < num_outputs; ++i) {
    if (masked && !(P.VU.elt<uint8_t>(0, i / 8) & (1 << (i % 8))))
      continue;
    int32_t acc = P.VU.elt<int32_t>(rd_num, i);
    for (int g = 0; g < 4; g++) {
      int8_t a = P.VU.elt<int8_t>(rs2_num, i * 4 + g);
      int8_t b = P.VU.elt<int8_t>(rs1_num, i * 4 + g);
      acc += (int32_t)a * (int32_t)b;
    }
    P.VU.elt<int32_t>(rd_num, i, true) = acc;
  }
  P.VU.vstart->write(0);
}
```

**Assembly test** (excerpt from directed test):
```asm
  # Test 1: All zeros -> expect 0
  vsetvli t0, t1, e8, m1, ta, ma
  # Load test vectors into v16, v24...
  .word 0x8388045b  # vdot4_i8i8_acc32 v8, v16, v24 (vm=1, unmasked)
  # Check: v8[0] == expected golden value
  li t3, 0x00000000
  bne t2, t3, fail
```

### 4. Validate the Output

```bash
python scripts/validate_output.py ./build/
```

This checks: directory structure exists, MATCH/MASK defines present, vstart
handling in execute body, `.word` encoding used (not `.insn r`), golden values
and bne comparisons present, and optionally assembles with riscv-gcc.

## Integrating with Spike

```bash
# 1. Clone Spike (or use the submodule)
git submodule update --init extern/spike

# 2. Generate your extension
rvxv generate --spec examples/int8_dot_product.yaml --output ./build/

# 3. Copy into the Spike source tree
cp build/spike/rvxv_encoding.h   extern/spike/riscv/
cp build/spike/insns/*.h         extern/spike/riscv/insns/
cp build/spike/extension.cc      extern/spike/riscv/rvxv_extension.cc

# 4. Build Spike with the extension
cd extern/spike && mkdir build && cd build
../configure --prefix=$RISCV && make -j$(nproc)

# 5. Run tests
spike --extension=rvxv path/to/test.elf
```

**Note:** The INT8 dot product workflow has been verified end-to-end (generate,
build into Spike, compile test, execute with correct results). Other operation
types have not been tested through this full integration workflow yet.
Requires `riscv64-linux-gnu-gcc` and a built Spike tree.

## Numeric Library (Standalone)

The numeric library works independently from the rest of RVXV:

```python
from rvxv.numeric import FP8E4M3, FP8E5M2, BFloat16, RoundingMode

# FP8 E4M3 -- no infinity, saturates to +/-448
fp8 = FP8E4M3()
bits = fp8.encode(1.5)        # -> 0x3E
value = fp8.decode(0x7E)      # -> 448.0 (max finite)
fp8.encode(float('inf'))      # -> 0x7E (saturates, no infinity)

# BFloat16 with configurable rounding
bf16_bits = BFloat16.from_fp32(3.14159, RoundingMode.RNE)
fp32_val = BFloat16.to_fp32(bf16_bits)

# ULP-aware comparison for verification
from rvxv.numeric import compare_with_tolerance, TOLERANCE_FMA
result = compare_with_tolerance(actual=1.0000001, expected=1.0, tolerance=TOLERANCE_FMA)
print(result.passed, result.ulp_dist)  # True, 1
```

## Supported Types and Operations

### Element Types

| Format | Bits | Range | Notes |
|--------|------|-------|-------|
| INT4 / UINT4 | 4 | [-8,7] / [0,15] | Packed sub-byte |
| INT8 / UINT8 | 8 | [-128,127] / [0,255] | Standard integer |
| INT16 / UINT16 | 16 | Standard | Standard integer |
| INT32 / UINT32 | 32 | Standard | Accumulator type |
| INT64 | 64 | Standard | Wide accumulator |
| FP8 E4M3 | 8 | +/-448 | No infinity, 2 NaN encodings |
| FP8 E5M2 | 8 | +/-57344 | IEEE 754-like |
| FP16 | 16 | IEEE 754 | Standard half-precision |
| BFloat16 | 16 | Same as FP32 | 5 rounding modes |
| FP32 | 32 | IEEE 754 | Via softfloat in Spike |
| FP64 | 64 | IEEE 754 | Double precision |
| MXFP8/6/4 | 4-8 | Varies | MX block floating point |

### Semantic Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `dot_product` | N-way dot product with accumulation | INT8x4 -> INT32 |
| `fma` | Fused multiply-add | BF16 * BF16 + FP32 |
| `mac` | Multiply-accumulate | INT4 * INT4 + INT32 |
| `multiply` | Element-wise multiply | Any type |
| `add` | Element-wise add | Any type |
| `fused_exp` | Fused exponential (for softmax) | BF16 -> BF16 |
| `convert` | Type conversion | FP8 -> BF16 |
| `compare` | Element-wise comparison | Any type |
| `outer_product` | Outer product (tiled matrix) | INT8 x INT8 -> INT32 |
| `reduction_sum` / `reduction_max` | Vector reductions | Any type |

## Writing Your Own Spec

### Encoding

RVXV supports two encoding formats:

```yaml
encoding:
  format: vector      # Recommended for vector AI instructions
  opcode: 0x5b        # custom-2 (0x0b, 0x2b, 0x5b, 0x7b are custom spaces)
  funct3: 0x0
  funct6: 0x20        # bits[31:26]; vm bit[25] handled automatically
```

Use `format: vector` for vector instructions. This correctly separates the vm
bit from funct6. Use `format: R-type` only for scalar custom instructions where
funct7 encodes all 7 bits [31:25].

### Operands

```yaml
operands:
  vs2: { type: vector, element: int8, groups: 4 }   # 4 elements per output
  vs1: { type: vector, element: int8, groups: 4 }
  vd:  { type: vector, element: int32 }              # wider accumulator
```

`groups: N` means N source elements contribute to one destination element
(e.g., 4-way dot product).

### Semantics

```yaml
semantics:
  operation: dot_product   # See table above
  accumulate: true         # Add to existing vd value
  saturation: none         # Or: signed, unsigned
```

## Pre-Built Presets

Presets are example specs showing how to describe common AI instruction patterns.
They are NOT official RISC-V extensions.

| Preset | Instructions | Description |
|--------|-------------|-------------|
| `common_ai_ops` | 5 | INT8 dot, BF16 FMA, FP8 dot, fused exp, INT4 MAC |
| `vme_zvbdot` | 4 | 4-element dot product variants |
| `ime_option_e` | 4 | Outer product matrix tile operations |
| `ame_outer_product` | 3 | Tiled matrix multiply-accumulate |

```bash
rvxv preset --list
rvxv preset --name common_ai_ops --output ./build/
```

## Testing

```bash
# Run all tests (207 tests)
pytest tests/ -v

# What the tests cover:
# - Numeric library: bit-accurate FP8/BF16/INT4/MX encode/decode,
#   cross-validated against struct.pack and IEEE 754 specs (102 tests)
# - Generators: structural validation of all generated artifacts (32 tests)
# - Core: spec parser, IR, type system, semantics engine (18 tests)
# - End-to-end: Spike E2E execution (3 tests), Spike artifacts (5 tests),
#   Verilator output validation (15 tests)
# - Golden values: independently computed expected values (13 tests)
# - Real toolchain: riscv-gcc assembly, g++ Spike compilation (13 tests)
# - CLI: all 4 commands work end-to-end (6 tests)

# Lint
ruff check src/ tests/ scripts/

# Validate generated output
python scripts/validate_output.py ./build/
```

### Git Submodules

For real-world integration testing, RVXV includes git submodules:

```bash
git submodule update --init --recursive

# This provides:
# extern/riscv-tests/  -- real riscv_test.h and test_macros.h headers
# extern/spike/        -- real Spike headers for compilation testing
```

When submodules are present, tests automatically use real headers instead of
minimal stubs.

## Project Structure

```
src/rvxv/
├── core/           # Spec parser, instruction IR, type system, semantics engine
├── numeric/        # Bit-accurate AI numeric types (FP8, BF16, INT4, MX formats)
├── generators/
│   ├── spike/      # Spike C++ extension generator (Jinja2 templates)
│   ├── tests/      # Assembly test generator (directed + random)
│   ├── assertions/ # SVA properties, RVFI checker, coverage, bind
│   └── docs/       # Markdown documentation generator
├── presets/        # Pre-built YAML specs for common AI instructions
└── cli.py          # Click CLI (generate, validate, preset, schema)

extern/
├── riscv-tests/    # Git submodule: riscv-tests framework headers
└── spike/          # Git submodule: Spike ISA simulator headers

scripts/
└── validate_output.py  # One-command output validation
```

## How to Extend

### Adding a New Operation Type

1. Add the operation to `SemanticOp` enum in `src/rvxv/core/type_system.py`
2. Create a Jinja2 template in `src/rvxv/generators/spike/templates/execute_body/`
3. Register it in `_TEMPLATE_MAP` in `src/rvxv/generators/spike/execute_gen.py`
4. Add test generation support in `src/rvxv/generators/tests/directed_gen.py`
5. Add golden value computation in `src/rvxv/core/semantics.py`

### Adding a New Element Type

1. Add the type to `ElementType` enum in `src/rvxv/core/type_system.py`
2. Add `TypeInfo` entry with width, signedness, and C type name
3. If it's a non-standard format, add encode/decode to `src/rvxv/numeric/`
4. Add conversion helpers to `_CONVERSION_IMPLS` in `spike_gen.py`
5. Add the type to `_TO_F32` / `_FROM_F32` in `execute_gen.py`

### Adding a New Generator

Implement the `Generator` base class in `src/rvxv/generators/base.py`:

```python
class MyGenerator(Generator):
    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        # Generate files, return list of created paths
        ...
```

## Comparison with Existing Tools

| | riscv-dv | riscv-formal | RVXV |
|---|---|---|---|
| Custom AI instructions | No | No | Yes (YAML-driven) |
| FP8/BF16/INT4 numerics | No | No | Bit-accurate |
| Spike extension gen | No | No | Yes (real API) |
| Assembly test gen | Constrained random | No | Directed + random |
| SVA properties | No | Base ISA only | Custom instruction templates |
| Open source | Yes | Yes | Yes (Apache 2.0) |

RVXV is complementary to these tools, not a replacement. riscv-dv is better for
general ISA-level random testing. riscv-formal is better for proving base ISA
correctness. RVXV fills the gap for custom AI instruction verification
infrastructure.

## License

Apache 2.0

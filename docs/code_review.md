# Code Review Notes

Generated artifacts were reviewed for correctness.

## Review Methodology

Generated files from `rvxv generate --spec examples/int8_dot_product.yaml`
were independently reviewed for correctness.

## Key Findings (Fixed)

1. **vm bit encoding (Fixed)**: Vector instructions now use `format: vector` with
   separate `funct6` field. The vm bit (bit 25) is correctly set: vm=1 for unmasked
   operations, vm=0 for masked operations. MASK does not include bit 25.

2. **Loop bounds (Fixed)**: Dot product execute body now iterates `vl / group_size`
   output elements instead of `vl` source elements, preventing out-of-bounds access.

3. **Assembly encoding (Fixed)**: Generated tests use `.word` directives instead of
   `.insn r`, which does not support vector register operands in GCC.

4. **Spike API (Fixed)**: Extension class uses `const` on `name()`, removed
   unnecessary `rocc.h` include, uses `decode.h` instead of `decode_macros.h`.

## Verified Correct

- Golden expected values are mathematically correct
- Vector vstart handling is implemented
- Mask register (v0) checking is implemented
- Test structure (load, execute, store, compare) is correct
- MATCH/MASK encoding constants are correct for the custom opcode space

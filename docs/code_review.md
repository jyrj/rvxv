# Verification Status

## End-to-End Spike Execution

All 7 example instruction specs pass end-to-end:
YAML spec -> Spike C++ extension -> compiled into Spike -> assembly test compiled
-> executed in Spike -> all golden value comparisons pass.

Tested specs: INT8 dot product, FP8 E4M3 dot product, INT4 packed MAC,
BF16 FMA, BF16-to-FP32 convert, fused exponential, INT32 reduction sum.

## Numeric Library

Exhaustively validated against Berkeley SoftFloat:
- FP8 E4M3: 256/256 bit patterns match
- FP8 E5M2: 256/256 bit patterns match
- BFloat16: 65536/65536 bit patterns match
- BFloat16 encode: 50000 random FP32 values across 5 rounding modes, 100% match

## Known Limitations

- Masked reduction tests are skipped (masked reductions have different
  semantics from element-wise masking and need separate golden value logic)
- SVA assertions are structural templates, not tested in simulation
- RVFI checker is a template requiring adaptation per trace format

"""Python reference model executor for instruction semantics.

Executes instruction semantics using the numeric library to produce
bit-accurate expected results for test generation.
"""

from __future__ import annotations

import math

import numpy as np

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import ElementType, SemanticOp, get_type_info


class SemanticsEngine:
    """Execute instruction semantics in Python to generate expected results.

    Uses the RVXV numeric library for bit-accurate computation of AI numeric
    types. Results are used by the test generator to produce expected values.
    """

    def execute(
        self,
        spec: InstructionSpec,
        operands: dict[str, np.ndarray],
        vl: int | None = None,
    ) -> np.ndarray:
        """Execute instruction semantics and return expected result.

        Args:
            spec: Instruction specification
            operands: Map of operand name to numpy array of raw bit patterns
            vl: Vector length (number of elements to process)

        Returns:
            numpy array of expected result bit patterns
        """
        match spec.semantics.operation:
            case SemanticOp.DOT_PRODUCT:
                return self._dot_product(spec, operands, vl)
            case SemanticOp.FMA:
                return self._fma(spec, operands, vl)
            case SemanticOp.MULTIPLY:
                return self._multiply(spec, operands, vl)
            case SemanticOp.ADD:
                return self._add(spec, operands, vl)
            case SemanticOp.REDUCTION_SUM:
                return self._reduction_sum(spec, operands, vl)
            case SemanticOp.REDUCTION_MAX:
                return self._reduction_max(spec, operands, vl)
            case SemanticOp.FUSED_EXP:
                return self._fused_exp(spec, operands, vl)
            case SemanticOp.CONVERT:
                return self._convert(spec, operands, vl)
            case SemanticOp.COMPARE:
                return self._compare(spec, operands, vl)
            case SemanticOp.OUTER_PRODUCT:
                return self._outer_product(spec, operands, vl)
            case SemanticOp.MAC:
                return self._mac(spec, operands, vl)
            case _:
                raise ValueError(f"Unsupported operation: {spec.semantics.operation}")

    def _decode_elements(
        self, raw: np.ndarray, element_type: ElementType
    ) -> np.ndarray:
        """Decode raw bit patterns to float64 values for computation."""
        if element_type == ElementType.INT8:
            return raw.astype(np.int8).astype(np.float64)
        elif element_type == ElementType.UINT8:
            return raw.astype(np.uint8).astype(np.float64)
        elif element_type == ElementType.INT4:
            # Decode two's complement 4-bit
            return np.where(raw & 0x8, raw.astype(np.int32) - 16, raw).astype(np.float64)
        elif element_type == ElementType.UINT4:
            return (raw & 0xF).astype(np.float64)
        elif element_type == ElementType.INT16:
            return raw.astype(np.int16).astype(np.float64)
        elif element_type == ElementType.INT32:
            return raw.astype(np.int32).astype(np.float64)
        elif element_type == ElementType.INT64:
            return raw.astype(np.int64).astype(np.float64)
        elif element_type in (ElementType.FP8_E4M3, ElementType.FP8_E5M2):
            from rvxv.numeric.fp8_e4m3 import FP8E4M3
            from rvxv.numeric.fp8_e5m2 import FP8E5M2

            fmt = FP8E4M3() if element_type == ElementType.FP8_E4M3 else FP8E5M2()
            return np.array([fmt.decode(int(b)) for b in raw], dtype=np.float64)
        elif element_type == ElementType.BF16:
            from rvxv.numeric.bfloat16 import BFloat16

            bf = BFloat16()
            return np.array([bf.decode(int(b)) for b in raw], dtype=np.float64)
        elif element_type == ElementType.FP16:
            return raw.astype(np.uint16).view(np.float16).astype(np.float64)
        elif element_type == ElementType.FP32:
            return raw.astype(np.uint32).view(np.float32).astype(np.float64)
        elif element_type == ElementType.FP64:
            return raw.astype(np.uint64).view(np.float64)
        else:
            # For MX and other types, treat as raw
            return raw.astype(np.float64)

    def _encode_result(self, values: np.ndarray, element_type: ElementType) -> np.ndarray:
        """Encode float64 values back to raw bit patterns."""
        if element_type in (ElementType.INT8, ElementType.UINT8):
            return values.astype(np.int32) & 0xFF
        elif element_type in (ElementType.INT4, ElementType.UINT4):
            return values.astype(np.int32) & 0xF
        elif element_type == ElementType.INT16:
            return values.astype(np.int32) & 0xFFFF
        elif element_type == ElementType.INT32:
            return values.astype(np.int64) & 0xFFFFFFFF
        elif element_type == ElementType.INT64:
            return values.astype(np.int64)
        elif element_type in (ElementType.FP8_E4M3, ElementType.FP8_E5M2):
            from rvxv.numeric.fp8_e4m3 import FP8E4M3
            from rvxv.numeric.fp8_e5m2 import FP8E5M2

            fmt = FP8E4M3() if element_type == ElementType.FP8_E4M3 else FP8E5M2()
            return np.array([fmt.encode(float(v)) for v in values], dtype=np.uint8)
        elif element_type == ElementType.BF16:
            from rvxv.numeric.bfloat16 import BFloat16

            return np.array([BFloat16.from_fp32(float(v)) for v in values], dtype=np.uint16)
        elif element_type == ElementType.FP32:
            with np.errstate(over="ignore"):
                return np.array(values, dtype=np.float32).view(np.uint32)
        else:
            return values.astype(np.int64)

    def _get_dest_type(self, spec: InstructionSpec) -> ElementType:
        """Get the destination element type."""
        dest = spec.dest_operands
        return next(iter(dest.values())).element

    def _get_source_types(self, spec: InstructionSpec) -> list[ElementType]:
        """Get source element types in order."""
        return [v.element for v in spec.source_operands.values()]

    def _get_group_size(self, spec: InstructionSpec) -> int:
        """Get element group size for dot product operations."""
        sources = list(spec.source_operands.values())
        if sources:
            return sources[0].groups
        return 1

    def _saturate(self, value: float, spec: InstructionSpec) -> float:
        """Apply saturation if specified."""
        if spec.semantics.saturation == "none":
            return value
        dest_type = self._get_dest_type(spec)
        info = get_type_info(dest_type)
        if spec.semantics.saturation == "signed":
            max_val = (1 << (info.width_bits - 1)) - 1
            min_val = -(1 << (info.width_bits - 1))
            return max(min_val, min(max_val, value))
        elif spec.semantics.saturation == "unsigned":
            max_val = (1 << info.width_bits) - 1
            return max(0, min(max_val, value))
        return value

    def _dot_product(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Compute element-group dot product."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)
        group_size = self._get_group_size(spec)

        a_raw = operands[sources[0]]
        b_raw = operands[sources[1]]
        a_vals = self._decode_elements(a_raw, src_types[0])
        b_vals = self._decode_elements(b_raw, src_types[1])

        # Determine number of output elements
        n_out = len(a_vals) // group_size
        if vl is not None:
            n_out = min(n_out, vl)

        results = np.zeros(n_out, dtype=np.float64)

        # Get accumulator initial values if accumulate mode
        dest_key = list(spec.dest_operands.keys())[0]
        if spec.semantics.accumulate and dest_key in operands:
            acc_raw = operands[dest_key]
            results[:len(acc_raw)] = self._decode_elements(acc_raw[:n_out], dest_type)

        for i in range(n_out):
            dot = 0.0
            for g in range(group_size):
                idx = i * group_size + g
                if idx < len(a_vals) and idx < len(b_vals):
                    dot += float(a_vals[idx]) * float(b_vals[idx])
            if spec.semantics.accumulate:
                results[i] += dot
            else:
                results[i] = dot
            results[i] = self._saturate(results[i], spec)

        return self._encode_result(results, dest_type)

    def _fma(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Fused multiply-accumulate: vd = (vs2 × vs1) + vd."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        a_vals = self._decode_elements(operands[sources[0]], src_types[0])
        b_vals = self._decode_elements(operands[sources[1]], src_types[1])

        n = min(len(a_vals), len(b_vals))
        if vl is not None:
            n = min(n, vl)

        results = np.zeros(n, dtype=np.float64)

        dest_key = list(spec.dest_operands.keys())[0]
        if spec.semantics.accumulate and dest_key in operands:
            acc_raw = operands[dest_key]
            acc = self._decode_elements(acc_raw[:n], dest_type)
            results[:len(acc)] = acc

        for i in range(n):
            results[i] = float(a_vals[i]) * float(b_vals[i]) + results[i]

        return self._encode_result(results, dest_type)

    def _multiply(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Element-wise multiply."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        a_vals = self._decode_elements(operands[sources[0]], src_types[0])
        b_vals = self._decode_elements(operands[sources[1]], src_types[1])

        n = min(len(a_vals), len(b_vals))
        if vl is not None:
            n = min(n, vl)

        results = a_vals[:n] * b_vals[:n]
        return self._encode_result(results, dest_type)

    def _add(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Element-wise add."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        a_vals = self._decode_elements(operands[sources[0]], src_types[0])
        b_vals = self._decode_elements(operands[sources[1]], src_types[1])

        n = min(len(a_vals), len(b_vals))
        if vl is not None:
            n = min(n, vl)

        results = a_vals[:n] + b_vals[:n]
        return self._encode_result(results, dest_type)

    def _reduction_sum(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Reduction sum: single scalar result = sum of all vector elements."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        vals = self._decode_elements(operands[sources[0]], src_types[0])
        n = len(vals) if vl is None else min(len(vals), vl)

        # Initial value from vs1[0] or accumulator
        init = 0.0
        if len(sources) > 1 and sources[1] in operands:
            init_raw = operands[sources[1]]
            if len(init_raw) > 0:
                init_type = src_types[1] if len(src_types) > 1 else dest_type
                init = float(self._decode_elements(init_raw[:1], init_type)[0])

        result = init + float(np.sum(vals[:n]))
        return self._encode_result(np.array([result]), dest_type)

    def _reduction_max(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Reduction max: single scalar result = max of all vector elements."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        vals = self._decode_elements(operands[sources[0]], src_types[0])
        n = len(vals) if vl is None else min(len(vals), vl)

        result = float(np.max(vals[:n]))
        return self._encode_result(np.array([result]), dest_type)

    def _fused_exp(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Fused exponential: vd[i] = exp(vs2[i]) — used for softmax."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        vals = self._decode_elements(operands[sources[0]], src_types[0])
        n = len(vals) if vl is None else min(len(vals), vl)

        def _safe_exp(v: float) -> float:
            if math.isnan(v):
                return float('nan')
            try:
                return math.exp(v)
            except OverflowError:
                return float('inf') if v > 0 else 0.0

        results = np.array([_safe_exp(float(v)) for v in vals[:n]])
        return self._encode_result(results, dest_type)

    def _convert(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Type conversion: convert elements from source to destination type."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        vals = self._decode_elements(operands[sources[0]], src_types[0])
        n = len(vals) if vl is None else min(len(vals), vl)

        return self._encode_result(vals[:n], dest_type)

    def _compare(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Element-wise comparison: result is mask (0 or 1)."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        a_vals = self._decode_elements(operands[sources[0]], src_types[0])
        b_vals = self._decode_elements(operands[sources[1]], src_types[1])

        n = min(len(a_vals), len(b_vals))
        if vl is not None:
            n = min(n, vl)

        # Default comparison is equality
        results = (a_vals[:n] == b_vals[:n]).astype(np.float64)
        return self._encode_result(results, dest_type)

    def _outer_product(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Outer product: result[i][j] = a[i] * b[j]."""
        sources = list(spec.source_operands.keys())
        src_types = self._get_source_types(spec)
        dest_type = self._get_dest_type(spec)

        a_vals = self._decode_elements(operands[sources[0]], src_types[0])
        b_vals = self._decode_elements(operands[sources[1]], src_types[1])

        result = np.outer(a_vals, b_vals).flatten()

        if spec.semantics.accumulate:
            dest_key = list(spec.dest_operands.keys())[0]
            if dest_key in operands:
                acc = self._decode_elements(operands[dest_key], dest_type)
                result[:len(acc)] += acc

        return self._encode_result(result, dest_type)

    def _mac(
        self, spec: InstructionSpec, operands: dict[str, np.ndarray], vl: int | None
    ) -> np.ndarray:
        """Multiply-accumulate: vd[i] += vs2[i] * vs1[i]."""
        return self._fma(spec, operands, vl)

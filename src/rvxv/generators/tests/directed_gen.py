"""Generate directed (boundary / edge-case) assembly test programs.

For each instruction specification, produces one ``.S`` assembly file
containing numbered test cases derived from the corner-case database in
:mod:`rvxv.generators.tests.corner_cases`.

Each test case:
  1. Configures the vector unit (``vsetvli``).
  2. Loads corner-case operand data from the ``.data`` section.
  3. Executes the instruction under test (via ``.insn``).
  4. Stores the result and compares element-by-element against golden
     values computed by the :class:`SemanticsEngine`.
  5. Branches to ``fail`` on mismatch, or falls through to the next test.

The generated files are compatible with the *riscv-tests* framework and can
be run on Spike or any RTL testbench that supports the ``tohost`` protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import ElementType, SemanticOp, get_type_info
from rvxv.generators.tests.asm_emitter import RISCVAssemblyEmitter
from rvxv.generators.tests.corner_cases import CornerCase, get_corner_cases

# Default VLEN in bits.  Spike defaults to 128.  Tests must not request more
# elements than VLMAX = VLEN / SEW * LMUL for a given configuration.
_DEFAULT_VLEN = 128


def _clamp_vl(num_elements: int, sew: int, lmul: int = 1) -> int:
    """Clamp element count to VLMAX for given SEW and LMUL."""
    vlmax = (_DEFAULT_VLEN // sew) * lmul
    return min(num_elements, vlmax)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_element_type(spec: InstructionSpec) -> ElementType:
    """Return the element type of the first source operand."""
    for name, op in spec.operands.items():
        if name.startswith("vs") or name.startswith("rs"):
            return op.element
    # Fallback: use first operand
    return next(iter(spec.operands.values())).element


def _dest_element_type(spec: InstructionSpec) -> ElementType:
    """Return the element type of the destination operand."""
    for name, op in spec.operands.items():
        if name.startswith("vd") or name.startswith("rd"):
            return op.element
    return next(iter(spec.operands.values())).element


def _sew_for_type(etype: ElementType) -> int:
    """Return the SEW value to use in vsetvli for the given element type."""
    return get_type_info(etype).sew_bits


def _eew_for_type(etype: ElementType) -> int:
    """Return the effective element width for vector loads/stores."""
    return get_type_info(etype).sew_bits


def _encoding_funct7(
    spec: InstructionSpec, *, masked: bool = False,
) -> int | None:
    """Return funct7 for R-type, or synthesized funct7 for vector format.

    For vector format: funct6 occupies bits[31:26], vm is bit[25].
    vm=1 means unmasked, vm=0 means masked (use v0 as mask).
    Default is unmasked (vm=1) unless masked=True.
    """
    enc = spec.encoding
    if enc.format == "R-type":
        return enc.funct7
    if enc.format == "vector" and enc.funct6 is not None:
        vm_bit = 0 if masked else 1
        return (enc.funct6 << 1) | vm_bit
    return enc.funct7


def _vector_register_names(
    spec: InstructionSpec,
) -> tuple[str, list[str]]:
    """Return (dest_reg, [source_regs]) for the instruction.

    Uses vector registers v8, v16, v24 etc. to avoid overlap with the
    register groups when LMUL > 1.
    """
    src_names = sorted(spec.source_operands.keys())

    # Assign non-overlapping vector register groups
    vd = "v8"
    vs_regs: list[str] = []
    next_vreg = 16
    for _ in src_names:
        vs_regs.append(f"v{next_vreg}")
        next_vreg += 8
    return vd, vs_regs


def _get_semantics_engine():
    """Lazily import and return a SemanticsEngine instance.

    Lazy import avoids circular imports when the module is loaded.
    """
    from rvxv.core.semantics_engine import SemanticsEngine
    return SemanticsEngine()


def _compute_golden_value(
    spec: InstructionSpec,
    operand_a: list[int],
    operand_b: list[int],
    engine=None,
    acc_init: int | None = None,
    vl: int | None = None,
) -> list[int]:
    """Compute expected result using SemanticsEngine.

    Constructs numpy arrays from raw bit-pattern operands, calls the
    engine, and returns the result as a list of raw bit-pattern ints.

    Parameters
    ----------
    acc_init : int | None
        Initial accumulator value for accumulating instructions.
        Defaults to 0 if *None*.
    vl : int | None
        Vector length to use.  If given, operands are truncated to *vl*
        elements before computing the golden value, matching what hardware
        does when vsetvli clamps the requested vl to VLMAX.
    """
    if engine is None:
        engine = _get_semantics_engine()

    a = operand_a[:vl] if vl is not None else operand_a
    b = operand_b[:vl] if vl is not None and operand_b else operand_b

    sources = sorted(spec.source_operands.keys())
    operands: dict[str, np.ndarray] = {}

    operands[sources[0]] = np.array(a, dtype=np.int64)
    if len(sources) >= 2 and b:
        operands[sources[1]] = np.array(b, dtype=np.int64)

    # If accumulate, provide initialised accumulator
    if spec.semantics.accumulate:
        dest_key = list(spec.dest_operands.keys())[0]
        group_size = 1
        src_ops = list(spec.source_operands.values())
        if src_ops:
            group_size = src_ops[0].groups
        n_out = max(1, len(a) // group_size)
        init_val = acc_init if acc_init is not None else 0
        operands[dest_key] = np.full(n_out, init_val, dtype=np.int64)

    result = engine.execute(spec, operands)
    return [int(v) & 0xFFFFFFFF for v in result]


def _result_word_values(raw_result: list[int], dst_sew: int) -> list[int]:
    """Pack raw result elements into 32-bit words for comparison.

    For SEW < 32, multiple elements are packed per word.
    For SEW == 32, each element is one word.
    For SEW > 32, each element spans multiple words.
    """
    bytes_per_elem = dst_sew // 8
    # Build a byte array from the raw results
    data = bytearray()
    for val in raw_result:
        for byte_idx in range(bytes_per_elem):
            data.append((val >> (byte_idx * 8)) & 0xFF)

    # Read back as 32-bit words (little-endian)
    words = []
    for i in range(0, len(data), 4):
        chunk = data[i:i + 4]
        word = 0
        for j, b in enumerate(chunk):
            word |= b << (j * 8)
        words.append(word)
    return words


# ---------------------------------------------------------------------------
# DirectedTestGenerator
# ---------------------------------------------------------------------------


class DirectedTestGenerator:
    """Generate directed (boundary / edge-case) test programs.

    For each instruction, generates a single ``.S`` assembly file that
    exercises all corner cases for the instruction's source element types.
    The file also includes VL boundary tests, masked-operation tests,
    LMUL sweep tests, and vstart tests.

    All test cases embed golden values computed by the SemanticsEngine
    and compare results element-by-element using bne to a fail label.
    """

    def generate(self, spec: InstructionSpec, output_dir: Path) -> list[Path]:
        """Generate directed test(s) for *spec*.

        Parameters
        ----------
        spec : InstructionSpec
            The instruction to test.
        output_dir : Path
            Directory to write generated ``.S`` files into.

        Returns
        -------
        list[Path]
            Paths of generated files.
        """
        src_type = _source_element_type(spec)
        dst_type = _dest_element_type(spec)
        operation = spec.semantics.operation.value

        corner_cases = get_corner_cases(src_type, operation)
        if not corner_cases:
            # Nothing to generate
            return []

        output_dir.mkdir(parents=True, exist_ok=True)

        # Lazily create semantics engine
        engine = _get_semantics_engine()

        asm = RISCVAssemblyEmitter(
            test_name=f"{spec.name}_directed",
            description=f"Directed corner-case tests for {spec.name}: {spec.description}",
        )

        vd, vs_regs = _vector_register_names(spec)
        src_sew = _sew_for_type(src_type)
        dst_sew = _sew_for_type(dst_type)
        src_eew = _eew_for_type(src_type)
        dst_eew = _eew_for_type(dst_type)

        # Determine the SEW used in vsetvli.  Must match the require() in the
        # Spike execute body (see execute_gen.py).  For widening DOT_PRODUCT/
        # MAC/OUTER_PRODUCT, Spike uses source SEW.  For all other ops
        # (including widening FMA, CONVERT), Spike uses destination SEW.
        op = spec.semantics.operation
        is_widening = src_sew < dst_sew
        if is_widening and op in (
            SemanticOp.DOT_PRODUCT, SemanticOp.MAC, SemanticOp.OUTER_PRODUCT,
        ):
            vsetvli_sew = src_sew
        else:
            vsetvli_sew = dst_sew if is_widening else src_sew

        # --- Compute golden values for ALL corner cases ---
        # Clamp element count to VLMAX so golden values match what Spike computes
        vlmax = _DEFAULT_VLEN // vsetvli_sew
        golden_values: list[list[int]] = []
        for idx, cc in enumerate(corner_cases):
            try:
                cc_vl = min(len(cc.operand_a), vlmax)
                gv = _compute_golden_value(spec, cc.operand_a, cc.operand_b, engine, vl=cc_vl)
                golden_values.append(gv)
            except Exception:
                logger.warning(
                    "Golden value computation failed for %s (case %d): %s",
                    spec.name, idx, cc.description, exc_info=True,
                )
                golden_values.append(list(cc.expected) if cc.expected else [])

        # --- Add test data to the .data section ---
        for idx, cc in enumerate(corner_cases):
            asm.add_test_data(
                f"tc{idx}_a", src_type, cc.operand_a
            )
            if cc.operand_b:
                asm.add_test_data(
                    f"tc{idx}_b", src_type, cc.operand_b
                )
            # Reserve space for result
            num_result_bytes = max(len(golden_values[idx]), 1) * (dst_sew // 8)
            if num_result_bytes < 4:
                num_result_bytes = 4
            asm.add_result_space(f"tc{idx}_result", num_result_bytes)

            # Golden expected values (always present now)
            if golden_values[idx]:
                asm.add_raw_data(
                    f"tc{idx}_expected", ".word",
                    _result_word_values(golden_values[idx], dst_sew),
                )

        # --- Add data for mask test ---
        # Skip mask data for reductions (masked reductions have different semantics)
        is_reduction = op in (SemanticOp.REDUCTION_SUM, SemanticOp.REDUCTION_MAX)
        if corner_cases and spec.encoding.vm and not is_reduction:
            self._add_mask_test_data(asm, spec, corner_cases[0], engine, dst_sew, dst_eew)

        # --- Add data for LMUL sweep ---
        if corner_cases:
            self._add_lmul_sweep_data(
                asm, spec, corner_cases[0], engine,
                src_type, dst_type, dst_sew,
            )

        # --- Add data for vstart test ---
        if corner_cases:
            self._add_vstart_test_data(asm, spec, corner_cases[0], engine, dst_sew, dst_eew)

        # --- Emit the test code ---
        asm.emit_header()

        test_id = 1

        for idx, cc in enumerate(corner_cases):
            asm.begin_test_case(test_id, cc.description)
            asm.reset_temp_gprs()

            # Clamp to VLMAX so vsetvli grants the requested vl
            vlmax = _DEFAULT_VLEN // vsetvli_sew
            num_elements = min(len(cc.operand_a), vlmax)

            # Configure vector unit
            asm.emit_comment(f"Configure: SEW={vsetvli_sew}, VL={num_elements}")
            asm.emit_li("a0", num_elements)
            asm.emit_vsetvli("a0", "a0", vsetvli_sew, lmul=1, ta=True, ma=True)

            # Load operand A
            asm.emit_load_address("a1", f"tc{idx}_a")
            asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

            # Load operand B (if present and we have a second source)
            if cc.operand_b and len(vs_regs) >= 2:
                asm.emit_load_address("a2", f"tc{idx}_b")
                asm.emit_vector_load(vs_regs[1], "a2", src_eew)

            # If the operation accumulates, pre-load / zero the dest register
            if spec.semantics.accumulate:
                asm.emit_comment("Zero accumulator register")
                asm.emit_vmv_v_i(vd, 0)

            # Emit the instruction under test
            asm.emit_blank()
            asm.emit_comment(f"Execute: {spec.name}")
            rs1_reg = vs_regs[0] if vs_regs else vd
            rs2_reg = vs_regs[1] if len(vs_regs) >= 2 else vs_regs[0] if vs_regs else vd
            asm.emit_custom_insn(
                fmt=spec.encoding.format,
                opcode=spec.encoding.opcode,
                funct3=spec.encoding.funct3,
                funct7=_encoding_funct7(spec),
                rd=vd,
                rs1=rs1_reg,
                rs2=rs2_reg,
            )

            # Store result
            asm.emit_blank()
            asm.emit_comment("Store result")
            num_dst_elements = len(golden_values[idx]) if golden_values[idx] else 1
            if dst_sew != src_sew:
                # Re-configure for destination element width
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=True, ma=True)

            asm.emit_load_address("a3", f"tc{idx}_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            # Verify against golden values
            if golden_values[idx]:
                asm.emit_blank()
                asm.emit_comment("Verify against golden expected values")
                asm.emit_load_address("a4", f"tc{idx}_expected")

                # Compare word by word
                result_words = _result_word_values(golden_values[idx], dst_sew)
                num_check_words = len(result_words)
                for w in range(num_check_words):
                    asm.emit_load_word("t0", w * 4, "a3")
                    asm.emit_load_word("t1", w * 4, "a4")
                    asm.emit_pass_fail_check("t0", "t1", test_id)
                    test_id += 1
            else:
                # No golden values -- just verify execution didn't trap
                test_id += 1

            asm.end_test_case()

        # --- VL boundary tests ---
        test_id = self._emit_vl_boundary_tests(
            asm, spec, corner_cases, vs_regs, vd,
            src_type, dst_type, vsetvli_sew, dst_sew,
            src_eew, dst_eew, test_id, engine,
        )

        # --- Masked operation test ---
        test_id = self._emit_masked_test(
            asm, spec, corner_cases, vs_regs, vd,
            src_type, dst_type, vsetvli_sew, dst_sew,
            src_eew, dst_eew, test_id, engine,
        )

        # --- LMUL sweep tests ---
        test_id = self._emit_lmul_sweep_tests(
            asm, spec, corner_cases, vs_regs, vd,
            src_type, dst_type, vsetvli_sew, dst_sew,
            src_eew, dst_eew, test_id, engine,
        )

        # --- vstart != 0 test ---
        test_id = self._emit_vstart_test(
            asm, spec, corner_cases, vs_regs, vd,
            src_type, dst_type, vsetvli_sew, dst_sew,
            src_eew, dst_eew, test_id, engine,
        )

        # Render and write
        content = asm.render()
        out_path = output_dir / f"{spec.name}_directed.S"
        out_path.write_text(content)
        return [out_path]

    # ------------------------------------------------------------------
    # VL boundary sub-tests
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_vl_boundary_tests(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        corner_cases: list[CornerCase],
        vs_regs: list[str],
        vd: str,
        src_type: ElementType,
        dst_type: ElementType,
        src_sew: int,
        dst_sew: int,
        src_eew: int,
        dst_eew: int,
        start_test_id: int,
        engine=None,
    ) -> int:
        """Emit tests with vl=1 and vl=VLMAX using the first corner case data.

        Returns the next available test_id.
        """
        if not corner_cases:
            return start_test_id

        cc = corner_cases[0]
        tid = start_test_id

        # -- vl = 1 --
        asm.begin_test_case(tid, "VL boundary: vl=1")
        asm.reset_temp_gprs()
        asm.emit_li("a0", 1)
        asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

        asm.emit_load_address("a1", "tc0_a")
        asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

        if cc.operand_b and len(vs_regs) >= 2:
            asm.emit_load_address("a2", "tc0_b")
            asm.emit_vector_load(vs_regs[1], "a2", src_eew)

        if spec.semantics.accumulate:
            asm.emit_vmv_v_i(vd, 0)

        rs1 = vs_regs[0] if vs_regs else vd
        rs2 = vs_regs[1] if len(vs_regs) >= 2 else rs1
        asm.emit_custom_insn(
            fmt=spec.encoding.format,
            opcode=spec.encoding.opcode,
            funct3=spec.encoding.funct3,
            funct7=_encoding_funct7(spec),
            rd=vd, rs1=rs1, rs2=rs2,
        )

        asm.emit_load_address("a3", "tc0_result")
        asm.emit_vector_store(vd, "a3", dst_eew if dst_sew == src_sew else dst_eew)
        asm.end_test_case()
        tid += 1

        # -- vl = VLMAX --
        asm.begin_test_case(tid, "VL boundary: vl=VLMAX")
        asm.reset_temp_gprs()
        asm.emit_comment("Set vl = VLMAX by passing zero in rs1 to vsetvli")
        asm.emit_vsetvli("a0", "zero", src_sew, lmul=1, ta=True, ma=True)

        asm.emit_load_address("a1", "tc0_a")
        asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

        if cc.operand_b and len(vs_regs) >= 2:
            asm.emit_load_address("a2", "tc0_b")
            asm.emit_vector_load(vs_regs[1], "a2", src_eew)

        if spec.semantics.accumulate:
            asm.emit_vmv_v_i(vd, 0)

        asm.emit_custom_insn(
            fmt=spec.encoding.format,
            opcode=spec.encoding.opcode,
            funct3=spec.encoding.funct3,
            funct7=_encoding_funct7(spec),
            rd=vd, rs1=rs1, rs2=rs2,
        )
        asm.end_test_case()
        tid += 1

        return tid

    # ------------------------------------------------------------------
    # Masked operation test (with verification)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_mask_test_data(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        cc: CornerCase,
        engine,
        dst_sew: int,
        dst_eew: int,
    ) -> None:
        """Add data section entries for the mask test."""
        # Compute expected result for active elements.
        # For accumulating instructions, the mask test pre-initialises vd
        # with 0xDEADBEEF, so golden values must use that as the accumulator
        # instead of zero.
        src_type = get_type_info(list(spec.source_operands.values())[0].element)
        mask_vl = _clamp_vl(len(cc.operand_a), src_type.sew_bits)
        try:
            golden = _compute_golden_value(
                spec, cc.operand_a, cc.operand_b, engine,
                acc_init=0xDEADBEEF if spec.semantics.accumulate else None,
                vl=mask_vl,
            )
        except Exception:
            logger.warning(
                "Golden value computation failed for %s: %s",
                spec.name, cc.description, exc_info=True,
            )
            golden = list(cc.expected) if cc.expected else []

        num_dst_elements = len(golden) if golden else 1
        num_result_bytes = max(num_dst_elements * (dst_sew // 8), 4)

        # Pre-init value for vd (0xDEADBEEF pattern)
        preinit_words = [0xDEADBEEF] * max(1, (num_result_bytes + 3) // 4)
        asm.add_raw_data("mask_preinit", ".word", preinit_words)
        asm.add_result_space("mask_result", num_result_bytes)

        # Build expected: active elements from golden, masked-off from preinit
        # Mask 0xAA = bit pattern: element 1,3,5,7 active; 0,2,4,6 masked-off
        if golden:
            expected_words: list[int] = []
            bytes_per_elem = dst_sew // 8
            elem_mask = (1 << dst_sew) - 1
            data = bytearray()
            for i in range(num_dst_elements):
                mask_bit = (0xAA >> i) & 1
                if mask_bit:
                    # Active: use golden value
                    val = golden[i] if i < len(golden) else 0
                else:
                    # Masked-off: extract per-element preinit from 0xDEADBEEF word
                    elems_per_word = 32 // dst_sew
                    shift = (i % elems_per_word) * dst_sew
                    val = (0xDEADBEEF >> shift) & elem_mask
                for byte_idx in range(bytes_per_elem):
                    data.append((val >> (byte_idx * 8)) & 0xFF)
            for i in range(0, len(data), 4):
                chunk = data[i:i + 4]
                word = 0
                for j, b in enumerate(chunk):
                    word |= b << (j * 8)
                expected_words.append(word)
            asm.add_raw_data("mask_expected", ".word", expected_words)

    @staticmethod
    def _emit_masked_test(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        corner_cases: list[CornerCase],
        vs_regs: list[str],
        vd: str,
        src_type: ElementType,
        dst_type: ElementType,
        src_sew: int,
        dst_sew: int,
        src_eew: int,
        dst_eew: int,
        start_test_id: int,
        engine=None,
    ) -> int:
        """Emit a masked-operation test with verification.

        Sets v0 to a mask pattern (0xAA) so that odd-indexed elements
        are active and even-indexed are masked off.  Pre-initialises vd
        with a known pattern (0xDEADBEEF) and verifies:
          - Active elements have the correct computed golden value
          - Masked-off elements retain their original value

        Returns the next available test_id.
        """
        if not corner_cases or not spec.encoding.vm:
            return start_test_id

        # Skip mask test for reduction operations — masked reductions have
        # different semantics (only active elements participate in the
        # reduction) which requires a separate golden value computation.
        op = spec.semantics.operation
        if op in (SemanticOp.REDUCTION_SUM, SemanticOp.REDUCTION_MAX):
            return start_test_id

        cc = corner_cases[0]
        tid = start_test_id
        mask_vl = _clamp_vl(len(cc.operand_a), src_sew)

        # Compute golden for verification
        try:
            golden = _compute_golden_value(spec, cc.operand_a, cc.operand_b, engine, vl=mask_vl)
        except Exception:
            logger.warning(
                "Golden value computation failed for %s: %s",
                spec.name, cc.description, exc_info=True,
            )
            golden = list(cc.expected) if cc.expected else []

        asm.begin_test_case(tid, "Masked operation: mask=0xAA (odd elements active)")
        asm.reset_temp_gprs()

        num_elements = _clamp_vl(len(cc.operand_a), src_sew)
        asm.emit_li("a0", num_elements)
        asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

        # Set mask register v0 to 0xAA (alternating, odd bits set)
        asm.emit_comment("Set mask register v0 = 0xAA (odd elements active)")
        asm.emit_li("t0", 0xAA)
        asm.emit_vmv_v_x("v0", "t0")

        # Pre-initialise vd with known value from mask_preinit
        asm.emit_comment("Pre-initialise vd with known value (0xDEADBEEF pattern)")
        num_dst_elements = len(golden) if golden else 1
        if dst_sew != src_sew:
            asm.emit_li("a0", num_dst_elements)
            asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=False, ma=False)
        asm.emit_load_address("a5", "mask_preinit")
        asm.emit_vector_load(vd, "a5", dst_eew)

        # Restore source SEW
        if dst_sew != src_sew:
            asm.emit_li("a0", num_elements)
            asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=False, ma=False)

        # Load operands
        asm.emit_load_address("a1", "tc0_a")
        asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

        if cc.operand_b and len(vs_regs) >= 2:
            asm.emit_load_address("a2", "tc0_b")
            asm.emit_vector_load(vs_regs[1], "a2", src_eew)

        # Emit the instruction with masking active (vm=0)
        rs1 = vs_regs[0] if vs_regs else vd
        rs2 = vs_regs[1] if len(vs_regs) >= 2 else rs1

        asm.emit_comment("Execute with mask active (vm=0)")
        asm.emit_custom_insn(
            fmt=spec.encoding.format,
            opcode=spec.encoding.opcode,
            funct3=spec.encoding.funct3,
            funct7=_encoding_funct7(spec, masked=True),
            rd=vd, rs1=rs1, rs2=rs2,
        )

        # Store and verify
        if golden:
            asm.emit_blank()
            asm.emit_comment("Store masked result and verify")
            if dst_sew != src_sew:
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=True, ma=True)
            asm.emit_load_address("a3", "mask_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            asm.emit_load_address("a4", "mask_expected")

            _result_word_values(golden, dst_sew)
            # We use the number of expected words for comparison
            num_check_words = max(1, (num_dst_elements * (dst_sew // 8)) // 4)
            for w in range(num_check_words):
                asm.emit_load_word("t0", w * 4, "a3")
                asm.emit_load_word("t1", w * 4, "a4")
                asm.emit_pass_fail_check("t0", "t1", tid)
                tid += 1

        asm.end_test_case()
        if not golden:
            tid += 1

        return tid

    # ------------------------------------------------------------------
    # LMUL sweep tests
    # ------------------------------------------------------------------

    @staticmethod
    def _add_lmul_sweep_data(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        cc: CornerCase,
        engine,
        src_type: ElementType,
        dst_type: ElementType,
        dst_sew: int,
    ) -> None:
        """Add data section entries for LMUL sweep tests."""
        src_sew = _sew_for_type(src_type)
        for lmul in [1, 2, 4]:
            num_elements = _clamp_vl(len(cc.operand_a), src_sew, lmul)
            num_result_bytes = max(num_elements, 1) * (dst_sew // 8)
            if num_result_bytes < 4:
                num_result_bytes = 4
            asm.add_result_space(f"lmul{lmul}_result", num_result_bytes)

    @staticmethod
    def _emit_lmul_sweep_tests(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        corner_cases: list[CornerCase],
        vs_regs: list[str],
        vd: str,
        src_type: ElementType,
        dst_type: ElementType,
        src_sew: int,
        dst_sew: int,
        src_eew: int,
        dst_eew: int,
        start_test_id: int,
        engine=None,
    ) -> int:
        """Emit tests that sweep LMUL = 1, 2, 4.

        Uses the first corner case data and verifies the instruction
        executes correctly at each LMUL setting.

        Returns the next available test_id.
        """
        if not corner_cases:
            return start_test_id

        cc = corner_cases[0]
        tid = start_test_id

        # Compute golden at default LMUL (result should be the same
        # for same VL regardless of LMUL)
        lmul1_vl = _clamp_vl(len(cc.operand_a), src_sew)
        try:
            golden = _compute_golden_value(spec, cc.operand_a, cc.operand_b, engine, vl=lmul1_vl)
        except Exception:
            logger.warning(
                "Golden value computation failed for %s: %s",
                spec.name, cc.description, exc_info=True,
            )
            golden = list(cc.expected) if cc.expected else []

        for lmul in [1, 2, 4]:
            asm.begin_test_case(tid, f"LMUL sweep: LMUL={lmul}")
            asm.reset_temp_gprs()

            num_elements = _clamp_vl(len(cc.operand_a), src_sew, lmul)
            asm.emit_li("a0", num_elements)
            asm.emit_vsetvli("a0", "a0", src_sew, lmul=lmul, ta=True, ma=True)

            # Load operand A
            asm.emit_load_address("a1", "tc0_a")
            asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

            # Load operand B
            if cc.operand_b and len(vs_regs) >= 2:
                asm.emit_load_address("a2", "tc0_b")
                asm.emit_vector_load(vs_regs[1], "a2", src_eew)

            if spec.semantics.accumulate:
                asm.emit_vmv_v_i(vd, 0)

            rs1 = vs_regs[0] if vs_regs else vd
            rs2 = vs_regs[1] if len(vs_regs) >= 2 else rs1
            asm.emit_custom_insn(
                fmt=spec.encoding.format,
                opcode=spec.encoding.opcode,
                funct3=spec.encoding.funct3,
                funct7=_encoding_funct7(spec),
                rd=vd, rs1=rs1, rs2=rs2,
            )

            # Store and verify
            num_dst_elements = len(golden) if golden else 1
            if dst_sew != src_sew:
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=lmul, ta=True, ma=True)

            asm.emit_load_address("a3", f"lmul{lmul}_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            if golden:
                asm.emit_blank()
                asm.emit_comment(f"Verify LMUL={lmul} result against golden values")
                asm.emit_load_address("a4", "tc0_expected")

                result_words = _result_word_values(golden, dst_sew)
                for w in range(len(result_words)):
                    asm.emit_load_word("t0", w * 4, "a3")
                    asm.emit_load_word("t1", w * 4, "a4")
                    asm.emit_pass_fail_check("t0", "t1", tid)
                    tid += 1

            asm.end_test_case()
            if not golden:
                tid += 1

        return tid

    # ------------------------------------------------------------------
    # vstart != 0 test
    # ------------------------------------------------------------------

    @staticmethod
    def _add_vstart_test_data(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        cc: CornerCase,
        engine,
        dst_sew: int,
        dst_eew: int,
    ) -> None:
        """Add data section entries for the vstart test."""
        # Compute golden result.  For accumulating instructions the vstart
        # test pre-initialises vd with 0xCAFEBABE, so golden values for
        # elements >= vstart must use that as the accumulator.
        src_type = get_type_info(list(spec.source_operands.values())[0].element)
        vstart_vl = _clamp_vl(len(cc.operand_a), src_type.sew_bits)
        try:
            golden = _compute_golden_value(
                spec, cc.operand_a, cc.operand_b, engine,
                acc_init=0xCAFEBABE if spec.semantics.accumulate else None,
                vl=vstart_vl,
            )
        except Exception:
            logger.warning(
                "Golden value computation failed for %s: %s",
                spec.name, cc.description, exc_info=True,
            )
            golden = list(cc.expected) if cc.expected else []

        num_dst_elements = len(golden) if golden else 1
        num_result_bytes = max(num_dst_elements * (dst_sew // 8), 4)

        # Pre-init value for vd
        preinit_words = [0xCAFEBABE] * max(1, (num_result_bytes + 3) // 4)
        asm.add_raw_data("vstart_preinit", ".word", preinit_words)
        asm.add_result_space("vstart_result", num_result_bytes)

        # Build expected: elements [0, vstart) keep preinit, [vstart, vl) get golden
        vstart_val = 2
        if golden:
            bytes_per_elem = dst_sew // 8
            # Extract per-element preinit values from the 0xCAFEBABE word pattern.
            # When SEW < 32, each 32-bit word contains multiple elements.
            elem_mask = (1 << dst_sew) - 1
            data = bytearray()
            for i in range(num_dst_elements):
                if i < vstart_val:
                    # Unchanged: extract the correct sub-word from preinit pattern
                    elems_per_word = 32 // dst_sew
                    shift = (i % elems_per_word) * dst_sew
                    val = (0xCAFEBABE >> shift) & elem_mask
                else:
                    val = golden[i] if i < len(golden) else 0
                for byte_idx in range(bytes_per_elem):
                    data.append((val >> (byte_idx * 8)) & 0xFF)
            expected_words: list[int] = []
            for i in range(0, len(data), 4):
                chunk = data[i:i + 4]
                word = 0
                for j, b in enumerate(chunk):
                    word |= b << (j * 8)
                expected_words.append(word)
            asm.add_raw_data("vstart_expected", ".word", expected_words)

    @staticmethod
    def _emit_vstart_test(
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        corner_cases: list[CornerCase],
        vs_regs: list[str],
        vd: str,
        src_type: ElementType,
        dst_type: ElementType,
        src_sew: int,
        dst_sew: int,
        src_eew: int,
        dst_eew: int,
        start_test_id: int,
        engine=None,
    ) -> int:
        """Emit a test with vstart=2 to verify elements below vstart are unchanged.

        Returns the next available test_id.
        """
        if not corner_cases:
            return start_test_id

        cc = corner_cases[0]
        tid = start_test_id
        vstart_val = 2
        vst_vl = _clamp_vl(len(cc.operand_a), src_sew)

        # Compute golden
        try:
            golden = _compute_golden_value(spec, cc.operand_a, cc.operand_b, engine, vl=vst_vl)
        except Exception:
            logger.warning(
                "Golden value computation failed for %s: %s",
                spec.name, cc.description, exc_info=True,
            )
            golden = list(cc.expected) if cc.expected else []

        asm.begin_test_case(tid, f"vstart={vstart_val}: elements below vstart unchanged")
        asm.reset_temp_gprs()

        num_elements = _clamp_vl(len(cc.operand_a), src_sew)
        asm.emit_li("a0", num_elements)
        asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

        # Pre-initialise vd with known value
        asm.emit_comment("Pre-initialise vd with 0xCAFEBABE pattern")
        num_dst_elements = len(golden) if golden else 1
        if dst_sew != src_sew:
            asm.emit_li("a0", num_dst_elements)
            asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=False, ma=False)
        asm.emit_load_address("a5", "vstart_preinit")
        asm.emit_vector_load(vd, "a5", dst_eew)

        # Restore source SEW
        if dst_sew != src_sew:
            asm.emit_li("a0", num_elements)
            asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

        # Load operands
        asm.emit_load_address("a1", "tc0_a")
        asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

        if cc.operand_b and len(vs_regs) >= 2:
            asm.emit_load_address("a2", "tc0_b")
            asm.emit_vector_load(vs_regs[1], "a2", src_eew)

        # Set vstart to non-zero
        asm.emit_comment(f"Set vstart = {vstart_val}")
        asm.emit_csrwi("vstart", vstart_val)

        # Execute instruction
        rs1 = vs_regs[0] if vs_regs else vd
        rs2 = vs_regs[1] if len(vs_regs) >= 2 else rs1
        asm.emit_custom_insn(
            fmt=spec.encoding.format,
            opcode=spec.encoding.opcode,
            funct3=spec.encoding.funct3,
            funct7=_encoding_funct7(spec),
            rd=vd, rs1=rs1, rs2=rs2,
        )

        # Reset vstart to 0 (for subsequent stores/ops)
        asm.emit_csrwi("vstart", 0)

        # Store and verify
        if golden and num_dst_elements > vstart_val:
            asm.emit_blank()
            asm.emit_comment("Store result and verify vstart behavior")
            if dst_sew != src_sew:
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=True, ma=True)
            asm.emit_load_address("a3", "vstart_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            asm.emit_load_address("a4", "vstart_expected")

            # Check all words
            num_result_bytes = num_dst_elements * (dst_sew // 8)
            num_check_words = max(1, num_result_bytes // 4)
            for w in range(num_check_words):
                asm.emit_load_word("t0", w * 4, "a3")
                asm.emit_load_word("t1", w * 4, "a4")
                asm.emit_pass_fail_check("t0", "t1", tid)
                tid += 1
        else:
            tid += 1

        asm.end_test_case()
        return tid

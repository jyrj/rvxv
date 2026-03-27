"""Constrained-random test generator for RISC-V assembly verification.

Generates assembly test programs with random operand values that are
constrained to the valid range of each element type.  A fixed seed ensures
deterministic, reproducible tests across runs.

Random tests complement the directed corner-case tests by exercising a wider
(but less targeted) portion of the input space.  Each test case embeds
golden expected values computed by the :class:`SemanticsEngine` and compares
results element-by-element for self-checking behaviour.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import ElementType, get_type_info
from rvxv.generators.tests.asm_emitter import RISCVAssemblyEmitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_range(element_type: ElementType) -> tuple[int, int]:
    """Return (min_raw, max_raw) inclusive range for raw bit patterns.

    For floating-point types the "range" is simply the full bit-pattern
    space (0 .. 2^width - 1).  The caller may further filter out NaN /
    Inf encodings if desired.
    """
    info = get_type_info(element_type)
    return 0, (1 << info.width_bits) - 1


def _random_values(
    rng: random.Random,
    element_type: ElementType,
    count: int,
    *,
    allow_special: bool = True,
) -> list[int]:
    """Generate *count* random bit-pattern values for *element_type*.

    When *allow_special* is ``False``, NaN and Inf encodings are excluded
    for floating-point types (useful for tests that need finite results).
    """
    info = get_type_info(element_type)
    lo, hi = _value_range(element_type)

    if not allow_special and info.is_float:
        return [_random_finite_fp(rng, element_type) for _ in range(count)]

    return [rng.randint(lo, hi) for _ in range(count)]


def _random_finite_fp(rng: random.Random, element_type: ElementType) -> int:
    """Generate a random *finite* floating-point bit pattern (no NaN / Inf)."""
    info = get_type_info(element_type)
    max_val = (1 << info.width_bits) - 1

    # Determine which bit patterns are NaN / Inf so we can reject them
    if element_type == ElementType.FP8_E4M3:
        # NaN: 0x7F, 0xFF
        reject = {0x7F, 0xFF}
    elif element_type == ElementType.FP8_E5M2:
        # Inf: 0x7C, 0xFC.  NaN: 0x7D..0x7F, 0xFD..0xFF
        reject = {0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF}
    elif element_type == ElementType.BF16:
        # Inf: 0x7F80, 0xFF80.  NaN: exp=0xFF, mantissa != 0
        reject = set()  # too many NaN encodings to enumerate; filter below
    elif element_type == ElementType.FP16:
        reject = set()
    else:
        reject = set()

    if reject:
        while True:
            val = rng.randint(0, max_val)
            if val not in reject:
                return val

    # For wider types, check structurally
    while True:
        val = rng.randint(0, max_val)
        if not _is_nan_or_inf(val, element_type):
            return val


def _is_nan_or_inf(bits: int, element_type: ElementType) -> bool:
    """Check whether *bits* encodes NaN or Inf for the given type."""
    info = get_type_info(element_type)
    if not info.is_float:
        return False

    if element_type == ElementType.FP8_E4M3:
        return (bits & 0x7F) == 0x7F  # both +NaN and -NaN
    elif element_type == ElementType.FP8_E5M2:
        exp = (bits >> 2) & 0x1F
        return exp == 0x1F  # all-ones exponent
    elif element_type == ElementType.BF16:
        exp = (bits >> 7) & 0xFF
        return exp == 0xFF
    elif element_type == ElementType.FP16:
        exp = (bits >> 10) & 0x1F
        return exp == 0x1F
    elif element_type == ElementType.FP32:
        exp = (bits >> 23) & 0xFF
        return exp == 0xFF
    return False


def _source_element_type(spec: InstructionSpec) -> ElementType:
    for name, op in spec.operands.items():
        if name.startswith("vs") or name.startswith("rs"):
            return op.element
    return next(iter(spec.operands.values())).element


def _dest_element_type(spec: InstructionSpec) -> ElementType:
    for name, op in spec.operands.items():
        if name.startswith("vd") or name.startswith("rd"):
            return op.element
    return next(iter(spec.operands.values())).element


def _sew_for_type(etype: ElementType) -> int:
    return get_type_info(etype).sew_bits


def _eew_for_type(etype: ElementType) -> int:
    return get_type_info(etype).sew_bits


def _encoding_funct7(
    spec: InstructionSpec, *, masked: bool = False,
) -> int | None:
    """vm=1 means unmasked, vm=0 means masked."""
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
    src_names = sorted(spec.source_operands.keys())
    vd = "v8"
    vs_regs: list[str] = []
    next_vreg = 16
    for _ in src_names:
        vs_regs.append(f"v{next_vreg}")
        next_vreg += 8
    return vd, vs_regs


def _get_semantics_engine():
    """Lazily import and return a SemanticsEngine instance."""
    from rvxv.core.semantics_engine import SemanticsEngine
    return SemanticsEngine()


def _compute_golden_value(
    spec: InstructionSpec,
    operand_a: list[int],
    operand_b: list[int],
    engine=None,
) -> list[int]:
    """Compute expected result using SemanticsEngine."""
    if engine is None:
        engine = _get_semantics_engine()

    sources = sorted(spec.source_operands.keys())
    operands: dict[str, np.ndarray] = {}

    operands[sources[0]] = np.array(operand_a, dtype=np.int64)
    if len(sources) >= 2 and operand_b:
        operands[sources[1]] = np.array(operand_b, dtype=np.int64)

    # If accumulate, provide zero-initialised accumulator
    if spec.semantics.accumulate:
        dest_key = list(spec.dest_operands.keys())[0]
        src_ops = list(spec.source_operands.values())
        group_size = src_ops[0].groups if src_ops else 1
        n_out = max(1, len(operand_a) // group_size)
        operands[dest_key] = np.zeros(n_out, dtype=np.int64)

    result = engine.execute(spec, operands)
    return [int(v) & 0xFFFFFFFF for v in result]


def _result_word_values(raw_result: list[int], dst_sew: int) -> list[int]:
    """Pack raw result elements into 32-bit words for comparison."""
    bytes_per_elem = dst_sew // 8
    data = bytearray()
    for val in raw_result:
        for byte_idx in range(bytes_per_elem):
            data.append((val >> (byte_idx * 8)) & 0xFF)

    words = []
    for i in range(0, len(data), 4):
        chunk = data[i:i + 4]
        word = 0
        for j, b in enumerate(chunk):
            word |= b << (j * 8)
        words.append(word)
    return words


# ---------------------------------------------------------------------------
# RandomTestGenerator
# ---------------------------------------------------------------------------


class RandomTestGenerator:
    """Generate constrained-random test programs.

    Produces a configurable number of random test iterations per instruction.
    Each iteration uses fresh random operand vectors while maintaining
    determinism through a controlled seed.  Golden values are computed
    by the SemanticsEngine and embedded for self-checking.

    Parameters
    ----------
    seed : int
        PRNG seed for reproducibility.
    num_tests : int
        Number of random test iterations per instruction.
    elements_per_test : int
        Number of vector elements per test iteration.
    """

    def __init__(
        self,
        seed: int = 42,
        num_tests: int = 100,
        elements_per_test: int = 16,
    ) -> None:
        self.seed = seed
        self.num_tests = num_tests
        self.elements_per_test = elements_per_test

    def generate(self, spec: InstructionSpec, output_dir: Path) -> list[Path]:
        """Generate a random-test ``.S`` file for *spec*.

        Parameters
        ----------
        spec : InstructionSpec
            The instruction to test.
        output_dir : Path
            Directory to write generated ``.S`` files into.

        Returns
        -------
        list[Path]
            Single-element list with the path of the generated file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        src_type = _source_element_type(spec)
        dst_type = _dest_element_type(spec)
        src_sew = _sew_for_type(src_type)
        dst_sew = _sew_for_type(dst_type)
        src_eew = _eew_for_type(src_type)
        dst_eew = _eew_for_type(dst_type)

        rng = random.Random(self.seed)

        # Lazily create semantics engine
        engine = _get_semantics_engine()

        asm = RISCVAssemblyEmitter(
            test_name=f"{spec.name}_random",
            description=(
                f"Constrained-random tests for {spec.name} "
                f"(seed={self.seed}, n={self.num_tests})"
            ),
        )

        vd, vs_regs = _vector_register_names(spec)
        num_sources = len(vs_regs)

        # ---- Generate data and golden values for all iterations ----
        golden_values: list[list[int]] = []

        for i in range(self.num_tests):
            a_vals = _random_values(rng, src_type, self.elements_per_test)
            asm.add_test_data(f"rand_{i}_a", src_type, a_vals)

            b_vals: list[int] = []
            if num_sources >= 2:
                b_vals = _random_values(rng, src_type, self.elements_per_test)
                asm.add_test_data(f"rand_{i}_b", src_type, b_vals)

            # Result space
            num_result_bytes = self.elements_per_test * (dst_sew // 8)
            if num_result_bytes < 4:
                num_result_bytes = 4
            asm.add_result_space(f"rand_{i}_result", num_result_bytes)

            # Compute golden values
            try:
                gv = _compute_golden_value(spec, a_vals, b_vals, engine)
                golden_values.append(gv)
                asm.add_raw_data(
                    f"rand_{i}_expected", ".word",
                    _result_word_values(gv, dst_sew),
                )
            except Exception:
                logger.warning(
                    "Golden value computation failed for %s (random iter %d)",
                    spec.name, i, exc_info=True,
                )
                golden_values.append([])

        # ---- Emit test code ----
        asm.emit_header()

        test_id = 1
        for i in range(self.num_tests):
            asm.begin_test_case(test_id, f"Random iteration {i} (seed={self.seed})")
            asm.reset_temp_gprs()

            # Configure vector unit
            asm.emit_li("a0", self.elements_per_test)
            asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

            # Load operand A
            asm.emit_load_address("a1", f"rand_{i}_a")
            asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

            # Load operand B
            if num_sources >= 2:
                asm.emit_load_address("a2", f"rand_{i}_b")
                asm.emit_vector_load(vs_regs[1], "a2", src_eew)

            # Zero accumulator if needed
            if spec.semantics.accumulate:
                asm.emit_vmv_v_i(vd, 0)

            # Execute instruction under test
            rs1 = vs_regs[0] if vs_regs else vd
            rs2 = vs_regs[1] if num_sources >= 2 else rs1
            asm.emit_custom_insn(
                fmt=spec.encoding.format,
                opcode=spec.encoding.opcode,
                funct3=spec.encoding.funct3,
                funct7=_encoding_funct7(spec),
                rd=vd,
                rs1=rs1,
                rs2=rs2,
            )

            # Store result
            num_dst_elements = len(golden_values[i]) if golden_values[i] else 1
            if dst_sew != src_sew:
                if num_dst_elements < 1:
                    num_dst_elements = 1
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=True, ma=True)

            asm.emit_load_address("a3", f"rand_{i}_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            # Verify against golden values
            if golden_values[i]:
                asm.emit_blank()
                asm.emit_comment("Verify against golden expected values")
                asm.emit_load_address("a4", f"rand_{i}_expected")

                result_words = _result_word_values(golden_values[i], dst_sew)
                for w in range(len(result_words)):
                    asm.emit_load_word("t0", w * 4, "a3")
                    asm.emit_load_word("t1", w * 4, "a4")
                    asm.emit_pass_fail_check("t0", "t1", test_id)
                    test_id += 1
            else:
                test_id += 1

            asm.end_test_case()

        # ---- Also generate a stress test with mixed VL values ----
        self._emit_vl_sweep(
            asm, spec, rng, src_type, dst_type,
            vs_regs, vd, src_sew, dst_sew, src_eew, dst_eew,
            start_test_id=test_id,
            engine=engine,
        )

        content = asm.render()
        out_path = output_dir / f"{spec.name}_random.S"
        out_path.write_text(content)
        return [out_path]

    # ------------------------------------------------------------------
    # VL sweep stress test
    # ------------------------------------------------------------------

    def _emit_vl_sweep(
        self,
        asm: RISCVAssemblyEmitter,
        spec: InstructionSpec,
        rng: random.Random,
        src_type: ElementType,
        dst_type: ElementType,
        vs_regs: list[str],
        vd: str,
        src_sew: int,
        dst_sew: int,
        src_eew: int,
        dst_eew: int,
        start_test_id: int,
        engine=None,
    ) -> None:
        """Emit a sweep over different VL values (1, 2, 4, 7, 8, 15, 16).

        Uses the same random data but varies VL to expose off-by-one bugs.
        Golden values are computed for each VL setting.
        """
        vl_values = [1, 2, 4, 7, 8, 15, 16]
        num_sources = len(vs_regs)

        # Add data for the VL sweep (use max VL worth of data)
        max_vl = max(vl_values)
        a_vals = _random_values(rng, src_type, max_vl)
        asm.add_test_data("vl_sweep_a", src_type, a_vals)

        b_vals: list[int] = []
        if num_sources >= 2:
            b_vals = _random_values(rng, src_type, max_vl)
            asm.add_test_data("vl_sweep_b", src_type, b_vals)

        result_bytes = max_vl * (dst_sew // 8)
        if result_bytes < 4:
            result_bytes = 4

        # Compute golden values for each VL
        vl_golden: dict[int, list[int]] = {}
        for vl in vl_values:
            a_sub = a_vals[:vl]
            b_sub = b_vals[:vl] if b_vals else []
            try:
                gv = _compute_golden_value(spec, a_sub, b_sub, engine)
                vl_golden[vl] = gv
            except Exception:
                logger.warning(
                    "Golden value computation failed for %s (VL sweep vl=%d)",
                    spec.name, vl, exc_info=True,
                )
                vl_golden[vl] = []

        for vl in vl_values:
            asm.add_result_space(f"vl_sweep_{vl}_result", result_bytes)
            if vl_golden[vl]:
                asm.add_raw_data(
                    f"vl_sweep_{vl}_expected", ".word",
                    _result_word_values(vl_golden[vl], dst_sew),
                )

        tid = start_test_id
        for vl in vl_values:
            asm.begin_test_case(tid, f"VL sweep: vl={vl}")
            asm.reset_temp_gprs()

            asm.emit_li("a0", vl)
            asm.emit_vsetvli("a0", "a0", src_sew, lmul=1, ta=True, ma=True)

            asm.emit_load_address("a1", "vl_sweep_a")
            asm.emit_vector_load(vs_regs[0] if vs_regs else vd, "a1", src_eew)

            if num_sources >= 2:
                asm.emit_load_address("a2", "vl_sweep_b")
                asm.emit_vector_load(vs_regs[1], "a2", src_eew)

            if spec.semantics.accumulate:
                asm.emit_vmv_v_i(vd, 0)

            rs1 = vs_regs[0] if vs_regs else vd
            rs2 = vs_regs[1] if num_sources >= 2 else rs1
            asm.emit_custom_insn(
                fmt=spec.encoding.format,
                opcode=spec.encoding.opcode,
                funct3=spec.encoding.funct3,
                funct7=_encoding_funct7(spec),
                rd=vd, rs1=rs1, rs2=rs2,
            )

            num_dst_elements = len(vl_golden[vl]) if vl_golden[vl] else 1
            if dst_sew != src_sew:
                if num_dst_elements < 1:
                    num_dst_elements = 1
                asm.emit_li("a0", num_dst_elements)
                asm.emit_vsetvli("a0", "a0", dst_sew, lmul=1, ta=True, ma=True)

            asm.emit_load_address("a3", f"vl_sweep_{vl}_result")
            asm.emit_vector_store(vd, "a3", dst_eew)

            # Verify
            if vl_golden[vl]:
                asm.emit_blank()
                asm.emit_comment(f"Verify VL={vl} against golden values")
                asm.emit_load_address("a4", f"vl_sweep_{vl}_expected")

                result_words = _result_word_values(vl_golden[vl], dst_sew)
                for w in range(len(result_words)):
                    asm.emit_load_word("t0", w * 4, "a3")
                    asm.emit_load_word("t1", w * 4, "a4")
                    asm.emit_pass_fail_check("t0", "t1", tid)
                    tid += 1
            else:
                tid += 1

            asm.end_test_case()

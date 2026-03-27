"""Generate RTL functional models and Verilator testbenches.

Produces a self-contained verification package for each instruction:
- SystemVerilog functional unit implementing the instruction logic
- SystemVerilog testbench with SVA assertion binds
- C++ Verilator testbench with embedded test vectors
- Makefile for Verilator compilation and execution
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import numpy as np

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.semantics_engine import SemanticsEngine
from rvxv.core.type_system import ElementType, SemanticOp, get_type_info
from rvxv.generators.base import Generator
from rvxv.generators.tests.corner_cases import get_corner_cases

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_src_element(spec: InstructionSpec) -> ElementType:
    """Get the source element type from the first source operand."""
    src_ops = list(spec.source_operands.values())
    return src_ops[0].element if src_ops else ElementType.INT8


def _get_dst_element(spec: InstructionSpec) -> ElementType:
    """Get the destination element type."""
    dst_ops = list(spec.dest_operands.values())
    return dst_ops[0].element if dst_ops else ElementType.INT32


def _build_test_vectors(spec: InstructionSpec, max_vectors: int = 16) -> list[dict]:
    """Build test vectors from corner cases with golden values.

    Returns a list of dicts with keys: name, operand_a, operand_b, expected.
    All values are lists of integers (bit patterns).
    """
    src_element = _get_src_element(spec)
    operation = spec.semantics.operation.value

    cases = get_corner_cases(src_element, operation)
    if not cases:
        return []

    engine = SemanticsEngine()
    src_names = list(spec.source_operands.keys())
    vectors = []

    for cc in cases[:max_vectors]:
        a = cc.operand_a
        b = cc.operand_b
        if not a or not b:
            continue

        try:
            operands: dict[str, np.ndarray] = {
                src_names[0]: np.array(a, dtype=np.int64),
                src_names[1]: np.array(b, dtype=np.int64),
            }
            result = engine.execute(spec, operands, vl=len(a))
            golden = [int(v) for v in result]
        except Exception:
            continue

        vectors.append({
            "name": cc.name,
            "operand_a": [int(v) for v in a],
            "operand_b": [int(v) for v in b],
            "expected": golden,
        })

    return vectors


class RTLGenerator(Generator):
    """Generate RTL functional models and Verilator testbenches."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate RTL + Verilator testbench for each instruction.

        Creates an ``rtl/`` subdirectory with per-instruction packages.
        """
        output_dir = output_dir / "rtl"
        self._ensure_dir(output_dir)

        generated: list[Path] = []

        for spec in specs:
            insn_dir = output_dir / spec.name
            self._ensure_dir(insn_dir)

            generated.append(self._gen_functional_unit(spec, insn_dir))
            generated.append(self._gen_testbench_sv(spec, insn_dir))
            generated.append(self._gen_verilator_tb(spec, insn_dir))
            generated.append(self._gen_makefile(spec, insn_dir))

        return generated

    def _build_context(self, spec: InstructionSpec) -> dict:
        """Build shared template context for a spec."""
        src_element = _get_src_element(spec)
        src_info = get_type_info(src_element)

        sources = []
        for name, op in spec.source_operands.items():
            info = get_type_info(op.element)
            sources.append({
                "name": name,
                "element": op.element,
                "width_bits": info.width_bits,
                "is_signed": info.is_signed,
                "is_float": info.is_float,
                "c_type": info.c_type,
                "groups": op.groups,
            })

        dest_name, dest_op = next(iter(spec.dest_operands.items()))
        dest_info_full = get_type_info(dest_op.element)

        group_size = 1
        src_ops = list(spec.source_operands.values())
        if src_ops and src_ops[0].groups > 1:
            group_size = src_ops[0].groups

        return {
            "name": spec.name,
            "name_upper": spec.name.upper(),
            "description": spec.description,
            "match_value": spec.encoding.match_value,
            "mask_value": spec.encoding.mask_value,
            "opcode": spec.encoding.opcode,
            "funct3": spec.encoding.funct3,
            "funct6": spec.encoding.funct6,
            "funct7": spec.encoding.funct7,
            "operation": spec.semantics.operation,
            "accumulate": spec.semantics.accumulate,
            "saturation": spec.semantics.saturation,
            "sources": sources,
            "dest_name": dest_name,
            "dest_element": dest_op.element,
            "dest_width": dest_info_full.width_bits,
            "dest_is_signed": dest_info_full.is_signed,
            "dest_is_float": dest_info_full.is_float,
            "src_width": src_info.width_bits,
            "src_is_signed": src_info.is_signed,
            "group_size": group_size,
            "is_dot_product": spec.semantics.operation in (
                SemanticOp.DOT_PRODUCT, SemanticOp.MAC,
            ),
            "is_reduction": spec.semantics.operation in (
                SemanticOp.REDUCTION_SUM, SemanticOp.REDUCTION_MAX,
            ),
        }

    def _gen_functional_unit(self, spec: InstructionSpec, out: Path) -> Path:
        """Generate the synthesizable functional unit module."""
        ctx = self._build_context(spec)
        template = self._env.get_template("functional_unit.sv.j2")
        path = out / f"{spec.name}.sv"
        path.write_text(template.render(**ctx))
        return path

    def _gen_testbench_sv(self, spec: InstructionSpec, out: Path) -> Path:
        """Generate SV testbench top with assertion binds."""
        ctx = self._build_context(spec)
        template = self._env.get_template("testbench.sv.j2")
        path = out / f"tb_{spec.name}.sv"
        path.write_text(template.render(**ctx))
        return path

    def _gen_verilator_tb(self, spec: InstructionSpec, out: Path) -> Path:
        """Generate C++ Verilator testbench with embedded test vectors."""
        ctx = self._build_context(spec)
        ctx["test_vectors"] = _build_test_vectors(spec)
        template = self._env.get_template("verilator_tb.cpp.j2")
        path = out / f"tb_{spec.name}.cpp"
        path.write_text(template.render(**ctx))
        return path

    def _gen_makefile(self, spec: InstructionSpec, out: Path) -> Path:
        """Generate Makefile for Verilator compilation."""
        ctx = self._build_context(spec)
        template = self._env.get_template("Makefile.j2")
        path = out / "Makefile"
        path.write_text(template.render(**ctx))
        return path

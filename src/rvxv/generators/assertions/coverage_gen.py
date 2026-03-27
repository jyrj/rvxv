"""Generate SystemVerilog functional coverage models for custom instructions.

Produces a single coverage module containing per-instruction covergroups with:
- Register address coverage
- Operand value range coverage (type-aware bins for int/float)
- Vector length (VL) and SEW coverage for vector instructions
- Cross coverage between operands and vector state
- Saturation event coverage
- Instruction mix coverage
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import get_type_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# SEW encoding: sew_bits -> vsew[2:0] encoding
_SEW_ENCODING: dict[int, int] = {
    8: 0b000,
    16: 0b001,
    32: 0b010,
    64: 0b011,
}

# LMUL encoding: lmul_value -> vlmul[2:0] encoding
# LMUL=1 -> 000, LMUL=2 -> 001, LMUL=4 -> 010, LMUL=8 -> 011
# LMUL=1/8 -> 101, LMUL=1/4 -> 110, LMUL=1/2 -> 111
_LMUL_ENCODING: dict[int, str] = {
    1: "000",
    2: "001",
    4: "010",
    8: "011",
}


def _build_source_info(spec: InstructionSpec) -> list[dict]:
    """Build source operand info for coverage template."""
    sources = []
    for name, op in spec.source_operands.items():
        info = get_type_info(op.element)
        if name.startswith("vs"):
            reg_name = name.replace("vs", "rs", 1)
        elif name.startswith("rs"):
            reg_name = name
        else:
            reg_name = name
        sources.append({
            "name": name,
            "reg_name": reg_name,
            "element": op.element,
            "element_name": op.element.value,
            "width_bits": info.width_bits,
            "is_float": info.is_float,
            "is_signed": info.is_signed,
            "reg_class": op.register_class,
            "groups": op.groups,
        })
    return sources


def _compute_sew_values(spec: InstructionSpec) -> tuple[list[int], dict[int, int]]:
    """Compute the allowed SEW values and their encodings."""
    min_sew = spec.constraints.min_sew
    max_sew = spec.constraints.max_sew

    if min_sew is None and max_sew is None:
        return [], {}

    sew_values = []
    sew_encoding = {}
    for bits, enc in sorted(_SEW_ENCODING.items()):
        if min_sew is not None and bits < min_sew:
            continue
        if max_sew is not None and bits > max_sew:
            continue
        sew_values.append(bits)
        sew_encoding[bits] = enc

    return sew_values, sew_encoding


class CoverageGenerator:
    """Generate SystemVerilog functional coverage model."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate the coverage module.

        Args:
            specs: List of instruction specifications.
            output_dir: Directory to write the generated coverage file.

        Returns:
            Single-element list containing the path to the coverage file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self._env.get_template("coverage.sv.j2")

        instructions = []
        for spec in specs:
            sources = _build_source_info(spec)

            dest_name, dest_op = next(iter(spec.dest_operands.items()))
            dest_info = get_type_info(dest_op.element)

            dest_reg_class = dest_op.register_class

            sew_values, sew_encoding = _compute_sew_values(spec)

            lmul_encoding = {}
            if spec.constraints.required_lmul:
                for lmul in spec.constraints.required_lmul:
                    if lmul in _LMUL_ENCODING:
                        lmul_encoding[lmul] = _LMUL_ENCODING[lmul]

            # Compute group size from first source operand (for dot products)
            first_source = next(iter(spec.source_operands.values()), None)
            group_size = first_source.groups if first_source else 1

            instructions.append({
                "name": spec.name,
                "match_value": spec.encoding.match_value,
                "mask_value": spec.encoding.mask_value,
                "sources": sources,
                "dest_reg_class": dest_reg_class,
                "dest_width_bits": dest_info.width_bits,
                "dest_is_float": dest_info.is_float,
                "dest_is_signed": dest_info.is_signed,
                "vl_dependent": spec.constraints.vl_dependent,
                "min_sew": spec.constraints.min_sew,
                "max_sew": spec.constraints.max_sew,
                "sew_values": sew_values,
                "sew_encoding": sew_encoding,
                "required_lmul": spec.constraints.required_lmul,
                "lmul_encoding": lmul_encoding,
                "saturation": spec.semantics.saturation,
                "accumulate": spec.semantics.accumulate,
                "group_size": group_size,
            })

        rendered = template.render(instructions=instructions)

        path = output_dir / "rvxv_coverage.sv"
        path.write_text(rendered)
        return [path]

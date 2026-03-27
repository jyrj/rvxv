"""Generate SystemVerilog Assertion modules for custom RISC-V instructions.

Each instruction gets a dedicated assertion module with properties for:
- Valid decode (encoding fields match)
- No spurious traps
- Destination register written
- Deterministic behavior (same inputs -> same outputs)
- Optional accumulation and saturation checks
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import get_type_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_source_info(spec: InstructionSpec) -> list[dict]:
    """Build source operand info dicts for template rendering."""
    sources = []
    for name, op in spec.source_operands.items():
        info = get_type_info(op.element)
        # Determine RVFI register name convention
        # vs1/vs2 -> rs1/rs2 in RVFI (RVFI uses rs1/rs2 for all source regs)
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


def _build_dest_info(spec: InstructionSpec) -> dict:
    """Build destination operand info dict for template rendering."""
    dest_name, dest_op = next(iter(spec.dest_operands.items()))
    info = get_type_info(dest_op.element)
    if dest_name.startswith("vd"):
        reg_name = "rd"
    elif dest_name.startswith("rd"):
        reg_name = dest_name
    else:
        reg_name = dest_name
    return {
        "name": dest_name,
        "reg_name": reg_name,
        "element": dest_op.element,
        "element_name": dest_op.element.value,
        "width_bits": info.width_bits,
        "is_float": info.is_float,
        "is_signed": info.is_signed,
        "reg_class": dest_op.register_class,
    }


def _has_vector_operands(spec: InstructionSpec) -> bool:
    """Check if any operand uses vector registers."""
    for op in spec.operands.values():
        if op.register_class == "vr":
            return True
    return False


class SVAGenerator:
    """Generate SystemVerilog Assertion modules for instruction verification."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate one SVA module per instruction.

        Args:
            specs: List of instruction specifications.
            output_dir: Directory to write generated SVA files.

        Returns:
            List of paths to generated files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self._env.get_template("sva_module.sv.j2")

        generated: list[Path] = []
        for spec in specs:
            path = self._generate_one(spec, template, output_dir)
            generated.append(path)
        return generated

    def _generate_one(
        self,
        spec: InstructionSpec,
        template: jinja2.Template,
        output_dir: Path,
    ) -> Path:
        """Render a single instruction's SVA module."""
        sources = _build_source_info(spec)
        dest = _build_dest_info(spec)
        enc = spec.encoding

        ctx = {
            "name": spec.name,
            "description": spec.description,
            "match_value": enc.match_value,
            "mask_value": enc.mask_value,
            "opcode": enc.opcode,
            "funct3": enc.funct3,
            "funct7": enc.funct7,
            "funct6": enc.funct6,
            "funct2": enc.funct2,
            "sources": sources,
            "dest": dest,
            "dest_width": dest["width_bits"],
            "accumulate": spec.semantics.accumulate,
            "saturation": spec.semantics.saturation,
            "has_vector_operands": _has_vector_operands(spec),
            "vl_dependent": spec.constraints.vl_dependent,
        }

        rendered = template.render(**ctx)
        path = output_dir / f"{spec.name}_assertions.sv"
        path.write_text(rendered)
        return path

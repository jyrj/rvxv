"""Generate Spike disassembly support for custom RISC-V instructions.

Produces a disasm.cc file that registers each instruction's mnemonic and
operand format with Spike's disassembler, plus a per-instruction disasm
entry for clean ``objdump``-style output.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec, OperandSpec

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Spike disassembler argument-class names for register operand types.
_REG_DISASM_ARG: dict[str, str] = {
    "vr": "arg_vd",   # vector register
    "gpr": "arg_rd",  # general-purpose register
    "fpr": "arg_fd",  # floating-point register
}

# Source register disasm argument classes (Spike uses different names for
# destination vs. source slots).
_SRC_DISASM_ARG: dict[tuple[str, int], str] = {
    ("vr", 1): "arg_vs1",
    ("vr", 2): "arg_vs2",
    ("vr", 3): "arg_vs3",
    ("gpr", 1): "arg_rs1",
    ("gpr", 2): "arg_rs2",
    ("fpr", 1): "arg_fs1",
    ("fpr", 2): "arg_fs2",
}


def _operand_disasm_arg(name: str, op: OperandSpec, src_index: int) -> str:
    """Return the Spike arg_t class to use for this operand slot."""
    rc = op.register_class
    if name.startswith("vd") or name.startswith("rd") or name.startswith("fd"):
        return _REG_DISASM_ARG.get(rc, "arg_vd")
    return _SRC_DISASM_ARG.get((rc, src_index), f"arg_vs{src_index}")


class DisasmGenerator:
    """Generate Spike disassembly registration code."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate disasm.cc from instruction specs.

        Args:
            specs: Instruction specifications.
            output_dir: Root spike output directory.

        Returns:
            Single-element list with the path to disasm.cc.
        """
        template = self._env.get_template("disasm.cc.j2")

        entries = []
        for spec in specs:
            entry = self._build_entry(spec)
            entries.append(entry)

        rendered = template.render(entries=entries)

        path = output_dir / "disasm.cc"
        path.write_text(rendered)
        return [path]

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _build_entry(self, spec: InstructionSpec) -> dict:
        """Build a single disasm entry dict for the template."""
        dests = spec.dest_operands
        sources = spec.source_operands

        # Build the operand display format: "vd, vs2, vs1"
        parts: list[str] = []
        disasm_args: list[str] = []

        # Destination operands first
        for name, op in dests.items():
            parts.append(name)
            disasm_args.append(_operand_disasm_arg(name, op, 0))

        # Source operands, numbered for Spike's slot system
        src_idx = 1
        for name, op in sources.items():
            parts.append(name)
            disasm_args.append(_operand_disasm_arg(name, op, src_idx))
            src_idx += 1

        disasm_format = ", ".join(parts)

        return {
            "name": spec.name,
            "name_upper": spec.name.upper(),
            "description": spec.description,
            "disasm_format": disasm_format,
            "disasm_args": disasm_args,
        }

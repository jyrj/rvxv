"""Generate a standalone Python RVFI trace checker script.

The generated script parses RVFI trace CSV files from RTL simulation and
validates retired custom instructions against reference values. It supports
both exact and tolerance-based comparison for floating-point results.
"""

from __future__ import annotations

import stat
from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import get_type_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class RVFICheckerGenerator:
    """Generate a standalone Python RVFI trace checker script."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate the RVFI trace checker Python script.

        Produces a single standalone Python script that knows how to match
        and validate all instructions defined in *specs*.

        Args:
            specs: List of instruction specifications.
            output_dir: Directory to write the generated checker script.

        Returns:
            Single-element list containing the path to the checker script.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self._env.get_template("rvfi_checker.py.j2")

        instructions = []
        for spec in specs:
            dest_name, dest_op = next(iter(spec.dest_operands.items()))
            dest_info = get_type_info(dest_op.element)

            # Build source operand names for golden model
            source_names = list(spec.source_operands.keys())

            instructions.append({
                "name": spec.name,
                "match_value": spec.encoding.match_value,
                "mask_value": spec.encoding.mask_value,
                "is_float_result": dest_info.is_float,
                "dest_width_bits": dest_info.width_bits,
                "is_signed": dest_info.is_signed,
                "has_accumulate": spec.semantics.accumulate,
                "saturation": spec.semantics.saturation,
                "source_names": source_names,
            })

        output_filename = "rvfi_checker.py"
        rendered = template.render(
            instructions=instructions,
            output_filename=output_filename,
        )

        path = output_dir / output_filename
        path.write_text(rendered)

        # Make the script executable
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        return [path]

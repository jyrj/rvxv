"""Generate instruction decode constants (MATCH/MASK) for Spike.

Produces an rvxv_encoding.h header with MATCH/MASK defines and DECLARE_INSN
macros for each custom instruction, following Spike's encoding conventions.
The filename uses 'rvxv_' prefix to avoid collision with Spike's own encoding.h.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class DecodeGenerator:
    """Generate instruction decode constants (MATCH/MASK) for Spike."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate rvxv_encoding.h from instruction specs.

        Args:
            specs: List of instruction specifications
            output_dir: Directory to write rvxv_encoding.h into

        Returns:
            Single-element list containing the path to rvxv_encoding.h
        """
        template = self._env.get_template("decode.h.j2")

        entries = []
        for spec in specs:
            entries.append(
                {
                    "name": spec.name,
                    "name_upper": spec.name.upper(),
                    "description": spec.description,
                    "match_value": spec.encoding.match_value,
                    "mask_value": spec.encoding.mask_value,
                }
            )

        rendered = template.render(entries=entries)

        path = output_dir / "rvxv_encoding.h"
        path.write_text(rendered)
        return [path]

"""YAML parser for instruction spec files.

Loads RISC-V instruction specifications from YAML files, validates them
against the Pydantic models, and provides schema export utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rvxv.core.instruction_ir import InstructionSpec


def _check_encoding_collisions(specs: list[InstructionSpec]) -> None:
    """Check for MATCH/MASK encoding collisions between instructions.

    Two instructions collide if their encodings overlap — i.e., there exists an
    instruction word that both instructions would decode.  This happens when:
        (MATCH_A & MASK_B) == (MATCH_B & MASK_B)  AND
        (MATCH_B & MASK_A) == (MATCH_A & MASK_A)

    Raises ValueError with details if any collision is detected.
    """
    if len(specs) < 2:
        return

    collisions: list[str] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            match_a, mask_a = a.encoding.match_value, a.encoding.mask_value
            match_b, mask_b = b.encoding.match_value, b.encoding.mask_value

            # Exact duplicate
            if match_a == match_b and mask_a == mask_b:
                collisions.append(
                    f"  '{a.name}' and '{b.name}' have identical encoding "
                    f"(MATCH=0x{match_a:08X}, MASK=0x{mask_a:08X})"
                )
            # Partial overlap: an instruction word could match both
            elif (match_a & mask_b) == (match_b & mask_b) or \
                 (match_b & mask_a) == (match_a & mask_a):
                collisions.append(
                    f"  '{a.name}' (MATCH=0x{match_a:08X}, MASK=0x{mask_a:08X}) "
                    f"overlaps with '{b.name}' (MATCH=0x{match_b:08X}, MASK=0x{mask_b:08X})"
                )

    if collisions:
        detail = "\n".join(collisions)
        raise ValueError(
            f"Encoding collision detected between instructions:\n{detail}\n"
            "Each instruction must have a unique MATCH/MASK encoding."
        )


def load_spec(path: str | Path) -> list[InstructionSpec]:
    """Load and validate instruction specs from a YAML file.

    Supports both single instruction (dict) and multiple instructions (list of dicts).
    Returns a list of validated InstructionSpec objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty spec file: {path}")

    # Normalize to list
    specs_data = raw if isinstance(raw, list) else [raw]

    specs = []
    for i, spec_data in enumerate(specs_data):
        try:
            specs.append(InstructionSpec.model_validate(spec_data))
        except ValidationError as e:
            # Add context about which spec failed
            location = f"instruction {i}" if len(specs_data) > 1 else "instruction"
            raise ValueError(f"Invalid {location} in {path}:\n{e}") from e

    _check_encoding_collisions(specs)

    return specs


def export_json_schema(output_path: str | Path) -> None:
    """Export JSON Schema for instruction specs (for editor autocompletion)."""
    schema = InstructionSpec.model_json_schema()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)


def validate_spec(spec_data: dict[str, Any]) -> InstructionSpec:
    """Validate a raw dictionary as an instruction spec."""
    return InstructionSpec.model_validate(spec_data)

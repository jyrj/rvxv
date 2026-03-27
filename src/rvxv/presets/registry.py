"""Preset discovery and loading for known RISC-V extension specifications."""

from __future__ import annotations

from pathlib import Path

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.spec_parser import load_spec

PRESETS_DIR = Path(__file__).parent / "specs"

PRESET_INFO = {
    "ime_option_e": {
        "file": "ime_option_e.yaml",
        "description": (
            "Custom outer product instructions (example configuration)"
        ),
    },
    "vme_zvbdot": {
        "file": "vme_zvbdot.yaml",
        "description": "Custom 4-element dot product instructions (example configuration)",
    },
    "ame_outer_product": {
        "file": "ame_outer_product.yaml",
        "description": (
            "RISC-V AME — tiled matrix outer product (example configuration)"
        ),
    },
    "common_ai_ops": {
        "file": "common_ai_ops.yaml",
        "description": (
            "Common AI ops: INT8 dot, BF16 FMA, FP8 dot, "
            "fused exp, INT4 MAC (example configuration)"
        ),
    },
}


def list_presets() -> dict[str, str]:
    """List available presets with descriptions."""
    return {name: info["description"] for name, info in PRESET_INFO.items()}


def load_preset(name: str) -> list[InstructionSpec]:
    """Load a preset by name."""
    if name not in PRESET_INFO:
        available = ", ".join(PRESET_INFO.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    spec_file = PRESETS_DIR / PRESET_INFO[name]["file"]
    if not spec_file.exists():
        raise FileNotFoundError(f"Preset file not found: {spec_file}")

    return load_spec(spec_file)

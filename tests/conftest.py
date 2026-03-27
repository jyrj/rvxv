"""Shared test fixtures for RVXV test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
PRESETS_DIR = Path(__file__).parent.parent / "src" / "rvxv" / "presets" / "specs"


@pytest.fixture
def int8_dot_spec_path() -> Path:
    return EXAMPLES_DIR / "int8_dot_product.yaml"


@pytest.fixture
def bf16_fma_spec_path() -> Path:
    return EXAMPLES_DIR / "bf16_fma.yaml"


@pytest.fixture
def fp8_e4m3_dot_spec_path() -> Path:
    return EXAMPLES_DIR / "fp8_e4m3_dot.yaml"


@pytest.fixture
def all_example_specs() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


@pytest.fixture
def all_preset_specs() -> list[Path]:
    return sorted(PRESETS_DIR.glob("*.yaml"))


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def int8_dot_spec_data() -> dict:
    return {
        "name": "vdot4_i8i8_acc32",
        "description": "4-way INT8 dot product with INT32 accumulation",
        "encoding": {
            "format": "R-type",
            "opcode": 0x5B,
            "funct3": 0x0,
            "funct7": 0x40,
        },
        "operands": {
            "vs2": {"type": "vector", "element": "int8", "groups": 4},
            "vs1": {"type": "vector", "element": "int8", "groups": 4},
            "vd": {"type": "vector", "element": "int32"},
        },
        "semantics": {
            "operation": "dot_product",
            "accumulate": True,
            "saturation": "none",
        },
        "constraints": {
            "vl_dependent": True,
            "mask_agnostic": True,
            "tail_agnostic": True,
        },
    }

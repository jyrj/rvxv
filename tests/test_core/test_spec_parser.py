"""Tests for spec parser and instruction IR."""

from __future__ import annotations

from pathlib import Path

import pytest

from rvxv.core.instruction_ir import (
    EncodingSpec,
    InstructionSpec,
)
from rvxv.core.spec_parser import export_json_schema, load_spec, validate_spec
from rvxv.core.type_system import SemanticOp


class TestEncodingSpec:
    def test_r_type_match_mask(self):
        enc = EncodingSpec(format="R-type", opcode=0x5B, funct3=0x0, funct7=0x40)
        # funct7=0x40 (0100000) at bits 31:25, funct3=0 at bits 14:12, opcode=0x5B at bits 6:0
        assert enc.match_value == (0x40 << 25) | (0x0 << 12) | 0x5B
        assert enc.mask_value == (0x7F << 25) | (0x7 << 12) | 0x7F

    def test_vector_match_mask(self):
        enc = EncodingSpec(format="vector", opcode=0x57, funct3=0x0, funct6=0x38)
        assert enc.match_value == (0x38 << 26) | (0x0 << 12) | 0x57
        assert enc.mask_value == (0x3F << 26) | (0x7 << 12) | 0x7F

    def test_r_type_requires_funct7(self):
        with pytest.raises(Exception):
            EncodingSpec(format="R-type", opcode=0x5B, funct3=0x0)

    def test_vector_requires_funct6(self):
        with pytest.raises(Exception):
            EncodingSpec(format="vector", opcode=0x57, funct3=0x0)

    def test_r4_type(self):
        enc = EncodingSpec(format="R4-type", opcode=0x43, funct3=0x0, funct2=0x1)
        assert enc.match_value == (0x1 << 25) | 0x43
        assert enc.mask_value == (0x3 << 25) | (0x7 << 12) | 0x7F


class TestInstructionSpec:
    def test_valid_dot_product(self, int8_dot_spec_data):
        spec = InstructionSpec.model_validate(int8_dot_spec_data)
        assert spec.name == "vdot4_i8i8_acc32"
        assert spec.semantics.operation == SemanticOp.DOT_PRODUCT
        assert spec.encoding.match_value > 0

    def test_invalid_no_source_operands(self):
        with pytest.raises(Exception):
            InstructionSpec.model_validate({
                "name": "bad_insn",
                "description": "missing sources for dot product",
                "encoding": {"format": "R-type", "opcode": 0x5B, "funct3": 0, "funct7": 0x40},
                "operands": {"vd": {"type": "vector", "element": "int32"}},
                "semantics": {"operation": "dot_product"},
            })

    def test_invalid_no_dest(self):
        with pytest.raises(Exception):
            InstructionSpec.model_validate({
                "name": "bad_insn",
                "description": "no destination",
                "encoding": {"format": "R-type", "opcode": 0x5B, "funct3": 0, "funct7": 0x40},
                "operands": {
                    "vs1": {"type": "vector", "element": "int8"},
                    "vs2": {"type": "vector", "element": "int8"},
                },
                "semantics": {"operation": "dot_product"},
            })

    def test_source_dest_properties(self, int8_dot_spec_data):
        spec = InstructionSpec.model_validate(int8_dot_spec_data)
        assert "vs1" in spec.source_operands
        assert "vs2" in spec.source_operands
        assert "vd" in spec.dest_operands

    def test_invalid_name(self):
        with pytest.raises(Exception):
            InstructionSpec.model_validate({
                "name": "INVALID-NAME",
                "description": "bad name",
                "encoding": {"format": "R-type", "opcode": 0x5B, "funct3": 0, "funct7": 0x40},
                "operands": {
                    "vs1": {"type": "vector", "element": "int8"},
                    "vs2": {"type": "vector", "element": "int8"},
                    "vd": {"type": "vector", "element": "int32"},
                },
                "semantics": {"operation": "add"},
            })


class TestSpecParser:
    def test_load_single_spec(self, int8_dot_spec_path):
        specs = load_spec(int8_dot_spec_path)
        assert len(specs) == 1
        assert specs[0].name == "vdot4_i8i8_acc32"

    def test_load_multi_spec(self, all_preset_specs):
        for path in all_preset_specs:
            specs = load_spec(path)
            assert len(specs) >= 1
            for spec in specs:
                assert spec.name
                assert spec.encoding.match_value >= 0

    def test_load_all_examples(self, all_example_specs):
        for path in all_example_specs:
            specs = load_spec(path)
            assert len(specs) >= 1

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_spec(Path("/nonexistent/spec.yaml"))

    def test_validate_spec(self, int8_dot_spec_data):
        spec = validate_spec(int8_dot_spec_data)
        assert spec.name == "vdot4_i8i8_acc32"

    def test_encoding_collision_detected(self, tmp_path):
        """Two instructions with identical encoding should raise ValueError."""
        import yaml

        specs_data = [
            {
                "name": "insn_a",
                "description": "test A",
                "encoding": {"format": "vector", "opcode": 0x5B, "funct3": 0x0, "funct6": 0x20},
                "operands": {
                    "vd": {"type": "vector", "element": "int32"},
                    "vs1": {"type": "vector", "element": "int8", "groups": 4},
                    "vs2": {"type": "vector", "element": "int8", "groups": 4},
                },
                "semantics": {"operation": "dot_product", "accumulate": True},
            },
            {
                "name": "insn_b",
                "description": "test B (same encoding as A)",
                "encoding": {"format": "vector", "opcode": 0x5B, "funct3": 0x0, "funct6": 0x20},
                "operands": {
                    "vd": {"type": "vector", "element": "int32"},
                    "vs1": {"type": "vector", "element": "int8", "groups": 4},
                    "vs2": {"type": "vector", "element": "int8", "groups": 4},
                },
                "semantics": {"operation": "dot_product", "accumulate": True},
            },
        ]
        spec_path = tmp_path / "collision.yaml"
        spec_path.write_text(yaml.dump(specs_data))
        with pytest.raises(ValueError, match="Encoding collision"):
            load_spec(spec_path)

    def test_no_collision_different_funct6(self, tmp_path):
        """Two instructions with different funct6 should load fine."""
        import yaml

        specs_data = [
            {
                "name": "insn_a",
                "description": "test A",
                "encoding": {"format": "vector", "opcode": 0x5B, "funct3": 0x0, "funct6": 0x20},
                "operands": {
                    "vd": {"type": "vector", "element": "int32"},
                    "vs1": {"type": "vector", "element": "int8", "groups": 4},
                    "vs2": {"type": "vector", "element": "int8", "groups": 4},
                },
                "semantics": {"operation": "dot_product", "accumulate": True},
            },
            {
                "name": "insn_b",
                "description": "test B (different funct6)",
                "encoding": {"format": "vector", "opcode": 0x5B, "funct3": 0x0, "funct6": 0x21},
                "operands": {
                    "vd": {"type": "vector", "element": "int32"},
                    "vs1": {"type": "vector", "element": "int8", "groups": 4},
                    "vs2": {"type": "vector", "element": "int8", "groups": 4},
                },
                "semantics": {"operation": "dot_product", "accumulate": True},
            },
        ]
        spec_path = tmp_path / "no_collision.yaml"
        spec_path.write_text(yaml.dump(specs_data))
        specs = load_spec(spec_path)
        assert len(specs) == 2

    def test_export_json_schema(self, tmp_path):
        schema_path = tmp_path / "schema.json"
        export_json_schema(schema_path)
        assert schema_path.exists()
        import json
        schema = json.loads(schema_path.read_text())
        assert "properties" in schema
        assert "name" in schema["properties"]

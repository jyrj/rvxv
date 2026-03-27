"""Tests for the CLI interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from rvxv.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_validate_example(self, runner):
        result = runner.invoke(main, ["validate", "--spec", "examples/int8_dot_product.yaml"])
        assert result.exit_code == 0
        assert "vdot4_i8i8_acc32" in result.output

    def test_validate_all_examples(self, runner):
        for path in Path("examples").glob("*.yaml"):
            result = runner.invoke(main, ["validate", "--spec", str(path)])
            assert result.exit_code == 0, f"Validation failed for {path}: {result.output}"

    def test_preset_list(self, runner):
        result = runner.invoke(main, ["preset", "--list"])
        assert result.exit_code == 0
        assert "common_ai_ops" in result.output

    def test_schema_export(self, runner, tmp_path):
        schema_path = tmp_path / "schema.json"
        result = runner.invoke(main, ["schema", "--output", str(schema_path)])
        assert result.exit_code == 0
        assert schema_path.exists()

    def test_generate_basic(self, runner, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(main, [
            "generate",
            "--spec", "examples/int8_dot_product.yaml",
            "--output", str(output_dir),
            "--targets", "spike,docs",
        ])
        assert result.exit_code == 0
        assert "Generated" in result.output

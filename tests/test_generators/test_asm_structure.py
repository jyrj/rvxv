"""Tests that generated assembly tests are self-checking."""
from pathlib import Path

from rvxv.core.spec_parser import load_spec
from rvxv.generators.tests.test_gen import AssemblyTestGenerator


class TestAsmStructure:
    def test_directed_has_golden_values(self, tmp_path):
        """Directed tests must include golden expected values."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        directed_files = list((tmp_path / "tests" / "directed").glob("*.S"))
        assert len(directed_files) > 0
        for f in directed_files:
            content = f.read_text()
            assert "expected" in content, f"No expected values in {f.name}"

    def test_directed_has_comparison(self, tmp_path):
        """Directed tests must compare results against expected."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        for f in (tmp_path / "tests" / "directed").glob("*.S"):
            content = f.read_text()
            # Should have branch-not-equal for checking
            assert "bne" in content, f"No comparison (bne) in {f.name}"

    def test_directed_has_pass_fail(self, tmp_path):
        """Directed tests must have pass/fail reporting."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        for f in (tmp_path / "tests" / "directed").glob("*.S"):
            content = f.read_text()
            assert "pass" in content.lower() or "PASS" in content
            assert "fail" in content.lower() or "FAIL" in content

    def test_directed_has_mask_test(self, tmp_path):
        """Directed tests must include masked operation test."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        for f in (tmp_path / "tests" / "directed").glob("*.S"):
            content = f.read_text()
            assert "mask" in content.lower(), f"No mask test in {f.name}"

    def test_directed_has_vstart_test(self, tmp_path):
        """Directed tests must include vstart test."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        for f in (tmp_path / "tests" / "directed").glob("*.S"):
            content = f.read_text()
            assert "vstart" in content, f"No vstart test in {f.name}"

    def test_random_has_golden_values(self, tmp_path):
        """Random tests must include golden expected values."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssemblyTestGenerator()
        gen.generate(specs, tmp_path)
        for f in (tmp_path / "tests" / "random").glob("*.S"):
            content = f.read_text()
            assert "expected" in content, f"No expected values in {f.name}"

    def test_all_examples_generate_tests(self, tmp_path):
        """All example specs should generate test output."""
        for yaml_path in Path("examples").glob("*.yaml"):
            out = tmp_path / yaml_path.stem
            specs = load_spec(yaml_path)
            gen = AssemblyTestGenerator()
            files = gen.generate(specs, out)
            assert len(files) >= 2, f"Too few files for {yaml_path.name}"

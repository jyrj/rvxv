"""Tests that generated SVA is structurally valid."""
from pathlib import Path

from rvxv.core.spec_parser import load_spec
from rvxv.generators.assertions.assertion_gen import AssertionGenerator


class TestSVAStructure:
    def test_sva_has_module_endmodule(self, tmp_path):
        """SVA files must have balanced module/endmodule."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssertionGenerator()
        gen.generate(specs, tmp_path)
        for sv in (tmp_path / "assertions").glob("*_assertions.sv"):
            content = sv.read_text()
            assert content.count("module ") >= 1
            assert content.count("endmodule") >= 1

    def test_sva_has_vector_ports(self, tmp_path):
        """SVA for vector instructions must have RVVI ports."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssertionGenerator()
        gen.generate(specs, tmp_path)
        for sv in (tmp_path / "assertions").glob("*_assertions.sv"):
            content = sv.read_text()
            assert "rvvi_vd_wdata" in content or "RVVI" in content, \
                f"No RVVI ports in {sv.name}"

    def test_sva_has_properties(self, tmp_path):
        """SVA must contain property definitions."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssertionGenerator()
        gen.generate(specs, tmp_path)
        for sv in (tmp_path / "assertions").glob("*_assertions.sv"):
            content = sv.read_text()
            assert "property" in content
            assert "endproperty" in content

    def test_coverage_has_vector_bins(self, tmp_path):
        """Coverage model must have vector-specific coverpoints."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssertionGenerator()
        gen.generate(specs, tmp_path)
        cov = (tmp_path / "assertions" / "rvxv_coverage.sv").read_text()
        # Should have VL and LMUL coverage
        assert "cp_vl" in cov
        assert "cp_vlmul" in cov

    def test_rvfi_checker_has_golden_model(self, tmp_path):
        """RVFI checker must have golden model support."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = AssertionGenerator()
        gen.generate(specs, tmp_path)
        checker = (tmp_path / "assertions" / "rvfi_checker.py").read_text()
        assert "SemanticsEngine" in checker or "golden" in checker.lower()

    def test_bind_uses_configurable_dut(self, tmp_path):
        """Bind file must use configurable DUT name."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        # Use custom DUT name
        gen = AssertionGenerator(dut_top="my_custom_core")
        gen.generate(specs, tmp_path)
        bind = (tmp_path / "assertions" / "rvxv_bind.sv").read_text()
        assert "my_custom_core" in bind
        assert "core_top" not in bind

    def test_presets_use_custom_opcodes(self):
        """All preset specs must use custom opcode spaces."""
        from rvxv.presets.registry import load_preset
        custom_opcodes = {0x0b, 0x2b, 0x5b, 0x7b}
        for preset_name in ["common_ai_ops", "vme_zvbdot", "ime_option_e", "ame_outer_product"]:
            specs = load_preset(preset_name)
            for spec in specs:
                assert spec.encoding.opcode in custom_opcodes, (
                    f"{preset_name}/{spec.name} uses opcode "
                    f"0x{spec.encoding.opcode:02x}, "
                    f"expected custom space"
                )

    def test_all_examples_generate_assertions(self, tmp_path):
        """All example specs should generate assertion output."""
        for yaml_path in Path("examples").glob("*.yaml"):
            out = tmp_path / yaml_path.stem
            specs = load_spec(yaml_path)
            gen = AssertionGenerator()
            files = gen.generate(specs, out)
            assert len(files) >= 3, f"Too few files for {yaml_path.name}"

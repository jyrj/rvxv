"""Tests that generated Spike C++ code is structurally valid."""
from pathlib import Path

from rvxv.core.spec_parser import load_spec
from rvxv.generators.spike.spike_gen import SpikeGenerator


class TestSpikeCompilable:
    def test_no_illegal_instruction(self, tmp_path):
        """Generated extension.cc must not use illegal_instruction handlers."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        ext_cc = (tmp_path / "spike" / "extension.cc").read_text()
        assert "illegal_instruction" not in ext_cc

    def test_has_real_function_pointers(self, tmp_path):
        """Generated extension.cc must have real handler function pointers."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        ext_cc = (tmp_path / "spike" / "extension.cc").read_text()
        assert "fast_vdot4_i8i8_acc32" in ext_cc

    def test_has_conversion_helper_bodies(self, tmp_path):
        """Generated code must have full conversion helper implementations."""
        specs = load_spec(Path("examples/fp8_e4m3_dot.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        ext_cc = (tmp_path / "spike" / "extension.cc").read_text()
        # Must have the function body, not just prototype
        assert "rvxv_fp8e4m3_to_f32" in ext_cc
        # Must have implementation (return statement, not just declaration)
        assert "return" in ext_cc.split("rvxv_fp8e4m3_to_f32")[1][:500]

    def test_includes_rvxv_encoding(self, tmp_path):
        """Generated extension.cc must include rvxv_encoding.h, not encoding.h."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        ext_cc = (tmp_path / "spike" / "extension.cc").read_text()
        assert "rvxv_encoding.h" in ext_cc

    def test_encoding_file_named_correctly(self, tmp_path):
        """Encoding file must be named rvxv_encoding.h."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        assert (tmp_path / "spike" / "rvxv_encoding.h").exists()
        assert not (tmp_path / "spike" / "encoding.h").exists()

    def test_execute_body_no_vi_vv_loop(self, tmp_path):
        """Generated execute bodies must not use the VI_VV_LOOP macro."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        for h_file in (tmp_path / "spike" / "insns").glob("*.h"):
            content = h_file.read_text()
            assert "VI_VV_LOOP" not in content, f"VI_VV_LOOP found in {h_file.name}"

    def test_execute_body_has_vstart(self, tmp_path):
        """Generated execute bodies must handle vstart."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        for h_file in (tmp_path / "spike" / "insns").glob("*.h"):
            content = h_file.read_text()
            assert "vstart" in content, f"No vstart handling in {h_file.name}"

    def test_execute_body_has_masking(self, tmp_path):
        """Generated execute bodies must handle masking."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        gen = SpikeGenerator()
        gen.generate(specs, tmp_path)
        for h_file in (tmp_path / "spike" / "insns").glob("*.h"):
            content = h_file.read_text()
            assert "masked" in content.lower() or "v_vm" in content, \
                f"No masking in {h_file.name}"

    def test_all_examples_generate(self, tmp_path):
        """All example specs should generate valid Spike output."""
        for yaml_path in Path("examples").glob("*.yaml"):
            out = tmp_path / yaml_path.stem
            specs = load_spec(yaml_path)
            gen = SpikeGenerator()
            files = gen.generate(specs, out)
            assert len(files) >= 4, f"Too few files for {yaml_path.name}"

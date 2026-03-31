"""End-to-end tests verifying generated artifacts are consistent."""
from pathlib import Path

import numpy as np

from rvxv.core.semantics_engine import SemanticsEngine
from rvxv.core.spec_parser import load_spec


class TestEndToEnd:
    def test_int8_dot_product_semantics(self):
        """Verify INT8 dot product semantics are correct."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        spec = specs[0]
        engine = SemanticsEngine()

        # 4-way dot product: [1,2,3,4] . [1,1,1,1] = 10
        a = np.array([1, 2, 3, 4], dtype=np.uint64)
        b = np.array([1, 1, 1, 1], dtype=np.uint64)
        sources = sorted(spec.source_operands.keys())
        result = engine.execute(spec, {sources[0]: a, sources[1]: b})
        assert result[0] == 10

    def test_int8_dot_product_max_values(self):
        """Verify INT8 dot product handles max values."""
        specs = load_spec(Path("examples/int8_dot_product.yaml"))
        spec = specs[0]
        engine = SemanticsEngine()

        # 4-way dot: [127,127,127,127] . [127,127,127,127] = 4*127*127 = 64516
        a = np.array([127, 127, 127, 127], dtype=np.uint64)
        b = np.array([127, 127, 127, 127], dtype=np.uint64)
        sources = sorted(spec.source_operands.keys())
        result = engine.execute(spec, {sources[0]: a, sources[1]: b})
        assert result[0] == 64516

    def test_full_pipeline_generates_all_artifacts(self, tmp_path):
        """Full pipeline should generate spike, tests, assertions, docs."""
        from rvxv.generators.assertions.assertion_gen import AssertionGenerator
        from rvxv.generators.docs.spec_doc_gen import DocGenerator
        from rvxv.generators.spike.spike_gen import SpikeGenerator
        from rvxv.generators.tests.test_gen import AssemblyTestGenerator

        specs = load_spec(Path("examples/int8_dot_product.yaml"))

        spike_files = SpikeGenerator().generate(specs, tmp_path)
        test_files = AssemblyTestGenerator().generate(specs, tmp_path)
        assert_files = AssertionGenerator().generate(specs, tmp_path)
        doc_files = DocGenerator().generate(specs, tmp_path)

        assert len(spike_files) >= 4
        assert len(test_files) >= 2
        assert len(assert_files) >= 3
        assert len(doc_files) >= 1

        # Verify key file contents
        ext_cc = (tmp_path / "spike" / "extension.cc").read_text()
        assert "illegal_instruction" not in ext_cc
        assert "extension_t" in ext_cc

        directed_s = list((tmp_path / "tests" / "directed").glob("*.S"))[0]
        content = directed_s.read_text()
        assert "expected" in content
        assert "bne" in content

    def test_preset_generates_multiple_instructions(self, tmp_path):
        """Preset with multiple instructions should generate per-instruction artifacts."""
        from rvxv.generators.spike.spike_gen import SpikeGenerator
        from rvxv.presets.registry import load_preset

        specs = load_preset("common_ai_ops")
        assert len(specs) == 5

        SpikeGenerator().generate(specs, tmp_path)
        insn_files = list((tmp_path / "spike" / "insns").glob("*.h"))
        assert len(insn_files) == 5

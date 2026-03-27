"""Main orchestrator for verification assertion generation.

Coordinates the four sub-generators to produce a complete verification suite:
- SVA assertion modules (per-instruction SystemVerilog properties)
- RVFI trace checker (standalone Python script)
- Coverage model (SystemVerilog functional coverage)
- Bind file (non-intrusive DUT integration)
"""

from __future__ import annotations

from pathlib import Path

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.generators.assertions.bind_gen import BindGenerator
from rvxv.generators.assertions.coverage_gen import CoverageGenerator
from rvxv.generators.assertions.rvfi_checker_gen import RVFICheckerGenerator
from rvxv.generators.assertions.sva_gen import SVAGenerator
from rvxv.generators.base import Generator


class AssertionGenerator(Generator):
    """Generate verification assertions and checkers.

    This is the top-level generator that orchestrates all assertion-related
    code generation. It delegates to four specialized sub-generators:

    1. SVAGenerator -- per-instruction SystemVerilog assertion modules
    2. RVFICheckerGenerator -- standalone Python RVFI trace checker
    3. CoverageGenerator -- SystemVerilog functional coverage model
    4. BindGenerator -- bind file for non-intrusive DUT integration
    """

    def __init__(
        self,
        dut_top: str = "core_top",
        clk_signal: str = "clk",
        rst_signal: str = "rst_n",
    ) -> None:
        self._dut_top = dut_top
        self._clk_signal = clk_signal
        self._rst_signal = rst_signal

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate all assertion and verification artifacts.

        Args:
            specs: List of instruction specifications to generate for.
            output_dir: Root output directory. An ``assertions/`` subdirectory
                will be created.

        Returns:
            List of paths to all generated files.
        """
        output_dir = output_dir / "assertions"
        self._ensure_dir(output_dir)

        generated: list[Path] = []

        sva = SVAGenerator()
        generated.extend(sva.generate(specs, output_dir))

        rvfi = RVFICheckerGenerator()
        generated.extend(rvfi.generate(specs, output_dir))

        coverage = CoverageGenerator()
        generated.extend(coverage.generate(specs, output_dir))

        bind = BindGenerator(
            dut_top=self._dut_top,
            clk_signal=self._clk_signal,
            rst_signal=self._rst_signal,
        )
        generated.extend(bind.generate(specs, output_dir))

        return generated

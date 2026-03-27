"""Main orchestrator for RISC-V assembly test suite generation.

Coordinates the :class:`DirectedTestGenerator` and
:class:`RandomTestGenerator` to produce a complete test suite for a set of
custom instruction specifications.  Output is written to a directory tree:

::

    <output_dir>/tests/
        directed/
            <insn>_directed.S   # one per instruction
        random/
            <insn>_random.S     # one per instruction

The generated ``.S`` files are self-contained riscv-tests-compatible programs
that can be assembled and run on Spike or any RTL testbench supporting the
``tohost`` pass/fail protocol.
"""

from __future__ import annotations

from pathlib import Path

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.generators.base import Generator
from rvxv.generators.tests.directed_gen import DirectedTestGenerator
from rvxv.generators.tests.random_gen import RandomTestGenerator


class TestGenerator(Generator):
    """Generate RISC-V assembly test suites.

    Produces both directed (corner-case) and constrained-random tests for
    every instruction in the provided specification list.

    Parameters
    ----------
    random_seed : int
        Seed for the constrained-random generator (default 42).
    random_count : int
        Number of random test iterations per instruction (default 100).
    """

    def __init__(
        self,
        random_seed: int = 42,
        random_count: int = 100,
    ) -> None:
        self.random_seed = random_seed
        self.random_count = random_count

    def generate(
        self, specs: list[InstructionSpec], output_dir: Path
    ) -> list[Path]:
        """Generate directed and random test suites for all *specs*.

        Parameters
        ----------
        specs : list[InstructionSpec]
            Instruction specifications to generate tests for.
        output_dir : Path
            Root output directory.  Tests are written under
            ``output_dir/tests/{directed,random}/``.

        Returns
        -------
        list[Path]
            Paths of all generated ``.S`` files.
        """
        tests_dir = output_dir / "tests"
        directed_dir = tests_dir / "directed"
        random_dir = tests_dir / "random"

        self._ensure_dir(directed_dir)
        self._ensure_dir(random_dir)

        generated: list[Path] = []

        directed_gen = DirectedTestGenerator()
        random_gen = RandomTestGenerator(
            seed=self.random_seed,
            num_tests=self.random_count,
        )

        for spec in specs:
            generated.extend(directed_gen.generate(spec, directed_dir))
            generated.extend(random_gen.generate(spec, random_dir))

        return generated

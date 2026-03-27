"""Abstract base class for all RVXV code generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rvxv.core.instruction_ir import InstructionSpec


class Generator(ABC):
    """Abstract base class for code generators."""

    @abstractmethod
    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate output files from instruction specs.

        Args:
            specs: List of instruction specifications
            output_dir: Directory to write generated files

        Returns:
            List of paths to generated files
        """
        ...

    def _ensure_dir(self, path: Path) -> None:
        """Ensure output directory exists."""
        path.mkdir(parents=True, exist_ok=True)

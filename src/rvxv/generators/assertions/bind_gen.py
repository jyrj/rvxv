"""Generate SystemVerilog bind file for non-intrusive assertion attachment.

The bind file connects assertion modules and the coverage model to the DUT
without modifying DUT source code. Users add this single file to their
simulation file list.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import get_type_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Default signal mapping -- can be overridden via constructor kwargs
_DEFAULT_DUT_TOP = "core_top"
_DEFAULT_CLK_SIGNAL = "clk"
_DEFAULT_RST_SIGNAL = "rst_n"
_DEFAULT_RVFI_PREFIX = "rvfi_"
_DEFAULT_VL_SIGNAL = "vl_value"
_DEFAULT_VSEW_SIGNAL = "vsew"
_DEFAULT_VLMUL_SIGNAL = "vlmul"
_DEFAULT_VM_SIGNAL = "rvvi_vm"


def _build_source_info(spec: InstructionSpec) -> list[dict]:
    """Build source operand info for bind template."""
    sources = []
    for name, op in spec.source_operands.items():
        info = get_type_info(op.element)
        if name.startswith("vs"):
            reg_name = name.replace("vs", "rs", 1)
        elif name.startswith("rs"):
            reg_name = name
        else:
            reg_name = name
        sources.append({
            "name": name,
            "reg_name": reg_name,
            "element_name": op.element.value,
            "width_bits": info.width_bits,
            "is_float": info.is_float,
            "is_signed": info.is_signed,
            "reg_class": op.register_class,
        })
    return sources


class BindGenerator:
    """Generate a SystemVerilog bind file for DUT integration.

    The bind file maps RVFI signals from the DUT to the assertion modules
    and coverage model. Signal names can be customized through constructor
    parameters.
    """

    def __init__(
        self,
        dut_top: str = _DEFAULT_DUT_TOP,
        clk_signal: str = _DEFAULT_CLK_SIGNAL,
        rst_signal: str = _DEFAULT_RST_SIGNAL,
        rvfi_prefix: str = _DEFAULT_RVFI_PREFIX,
        vl_signal: str = _DEFAULT_VL_SIGNAL,
        vsew_signal: str = _DEFAULT_VSEW_SIGNAL,
        vlmul_signal: str = _DEFAULT_VLMUL_SIGNAL,
        vm_signal: str = _DEFAULT_VM_SIGNAL,
    ) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._dut_top = dut_top
        self._clk_signal = clk_signal
        self._rst_signal = rst_signal
        self._rvfi_prefix = rvfi_prefix
        self._vl_signal = vl_signal
        self._vsew_signal = vsew_signal
        self._vlmul_signal = vlmul_signal
        self._vm_signal = vm_signal

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate the bind file.

        Args:
            specs: List of instruction specifications.
            output_dir: Directory to write the generated bind file.

        Returns:
            Single-element list containing the path to the bind file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self._env.get_template("bind.sv.j2")

        instructions = []
        for spec in specs:
            sources = _build_source_info(spec)

            dest_name, dest_op = next(iter(spec.dest_operands.items()))
            if dest_name.startswith("vd"):
                dest_reg_name = "rd"
            elif dest_name.startswith("rd"):
                dest_reg_name = dest_name
            else:
                dest_reg_name = dest_name

            # Check if any operand uses vector registers
            has_vector_operands = any(
                op.register_class == "vr" for op in spec.operands.values()
            )

            instructions.append({
                "name": spec.name,
                "sources": sources,
                "dest_reg_name": dest_reg_name,
                "has_vector_operands": has_vector_operands,
            })

        rendered = template.render(
            instructions=instructions,
            dut_top=self._dut_top,
            clk_signal=self._clk_signal,
            rst_signal=self._rst_signal,
            rvfi_prefix=self._rvfi_prefix,
            vl_signal=self._vl_signal,
            vsew_signal=self._vsew_signal,
            vlmul_signal=self._vlmul_signal,
            vm_signal=self._vm_signal,
        )

        path = output_dir / "rvxv_bind.sv"
        path.write_text(rendered)
        return [path]

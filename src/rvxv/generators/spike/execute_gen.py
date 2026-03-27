"""Generate Spike execute functions for custom RISC-V instructions.

Each instruction gets an ``insns/<name>.h`` file containing the execute body
that Spike includes inside its generated ``execute_insn`` wrapper.  The body
implements vector loops, type conversions, and the core arithmetic for every
supported SemanticOp variant.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from rvxv.core.instruction_ir import InstructionSpec
from rvxv.core.type_system import ElementType, SemanticOp, TypeInfo, get_type_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Mapping from SemanticOp to the Jinja2 sub-template that renders its body
# ---------------------------------------------------------------------------
_OP_TEMPLATE: dict[SemanticOp, str] = {
    SemanticOp.DOT_PRODUCT: "execute_body/dot_product.cc.j2",
    SemanticOp.FMA: "execute_body/fma.cc.j2",
    SemanticOp.MULTIPLY: "execute_body/fma.cc.j2",  # multiply reuses fma template
    SemanticOp.ADD: "execute_body/fma.cc.j2",  # add reuses fma template
    SemanticOp.MAC: "execute_body/fma.cc.j2",  # mac reuses fma template
    SemanticOp.REDUCTION_SUM: "execute_body/reduction.cc.j2",
    SemanticOp.REDUCTION_MAX: "execute_body/reduction.cc.j2",
    SemanticOp.FUSED_EXP: "execute_body/fma.cc.j2",  # fused_exp uses fma structure
    SemanticOp.CONVERT: "execute_body/convert.cc.j2",
    SemanticOp.COMPARE: "execute_body/compare.cc.j2",
    SemanticOp.OUTER_PRODUCT: "execute_body/dot_product.cc.j2",  # outer_product variant
}

# ---------------------------------------------------------------------------
# AI-type conversion helper names
# ---------------------------------------------------------------------------
_TO_F32: dict[ElementType, str] = {
    ElementType.FP8_E4M3: "rvxv_fp8e4m3_to_f32",
    ElementType.FP8_E5M2: "rvxv_fp8e5m2_to_f32",
    ElementType.FP16: "rvxv_fp16_to_f32",
    ElementType.BF16: "rvxv_bf16_to_f32",
    ElementType.FP32: "rvxv_softfloat_to_f32",
    ElementType.MXFP8: "rvxv_mxfp8_to_f32",
    ElementType.MXFP6_E3M2: "rvxv_mxfp6e3m2_to_f32",
    ElementType.MXFP6_E2M3: "rvxv_mxfp6e2m3_to_f32",
    ElementType.MXFP4: "rvxv_mxfp4_to_f32",
}

_FROM_F32: dict[ElementType, str] = {
    ElementType.FP8_E4M3: "rvxv_f32_to_fp8e4m3",
    ElementType.FP8_E5M2: "rvxv_f32_to_fp8e5m2",
    ElementType.FP16: "rvxv_f32_to_fp16",
    ElementType.BF16: "rvxv_f32_to_bf16",
    ElementType.FP32: "rvxv_f32_to_softfloat",
    ElementType.MXFP8: "rvxv_f32_to_mxfp8",
    ElementType.MXFP6_E3M2: "rvxv_f32_to_mxfp6e3m2",
    ElementType.MXFP6_E2M3: "rvxv_f32_to_mxfp6e2m3",
    ElementType.MXFP4: "rvxv_f32_to_mxfp4",
}

# SEW enum names used by Spike for require() checks
_SEW_ENUM: dict[int, str] = {
    8: "e8",
    16: "e16",
    32: "e32",
    64: "e64",
}


def _needs_conversion(etype: ElementType) -> bool:
    """Return True if the element type requires a software conversion helper."""
    return etype in _TO_F32


def _compute_ratio(src_info: TypeInfo, dst_info: TypeInfo) -> int:
    """Number of source elements that map to one destination element.

    For dot-products this is dst_width / src_width (e.g. 32/8 = 4).
    """
    if src_info.width_bits == 0:
        return 1
    return max(1, dst_info.width_bits // src_info.width_bits)


class ExecuteGenerator:
    """Generate Spike ``insns/<name>.h`` execute bodies."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Register helpers available inside all templates
        self._env.globals.update(
            {
                "SemanticOp": SemanticOp,
                "needs_conversion": _needs_conversion,
                "to_f32_func": lambda et: _TO_F32.get(et, ""),
                "from_f32_func": lambda et: _FROM_F32.get(et, ""),
                "sew_enum": lambda bits: _SEW_ENUM.get(bits, f"e{bits}"),
            }
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def generate(self, specs: list[InstructionSpec], output_dir: Path) -> list[Path]:
        """Generate one ``insns/<name>.h`` per instruction.

        Args:
            specs: Instruction specifications to generate execute bodies for.
            output_dir: Root spike output directory (insns/ subdirectory is used).

        Returns:
            List of paths to generated files.
        """
        insns_dir = output_dir / "insns"
        insns_dir.mkdir(parents=True, exist_ok=True)

        generated: list[Path] = []
        for spec in specs:
            path = self._generate_one(spec, insns_dir)
            generated.append(path)
        return generated

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _generate_one(self, spec: InstructionSpec, insns_dir: Path) -> Path:
        """Render a single instruction's execute body."""
        ctx = self._build_context(spec)

        template = self._env.get_template("execute.cc.j2")
        rendered = template.render(**ctx)

        path = insns_dir / f"{spec.name}.h"
        path.write_text(rendered)
        return path

    def _build_context(self, spec: InstructionSpec) -> dict:
        """Build the full Jinja2 template context for *spec*."""
        sources = spec.source_operands
        dests = spec.dest_operands
        op = spec.semantics.operation

        # Pick the first source pair and destination for the common case
        src_names = list(sources.keys())
        dst_name = list(dests.keys())[0]
        dst_op = dests[dst_name]
        dst_info = get_type_info(dst_op.element)

        # Source type info
        src_infos: list[dict] = []
        for sn in src_names:
            so = sources[sn]
            si = get_type_info(so.element)
            src_infos.append(
                {
                    "name": sn,
                    "reg_field": sn,
                    "element": so.element,
                    "c_type": si.c_type,
                    "width_bits": si.width_bits,
                    "is_float": si.is_float,
                    "needs_conv": _needs_conversion(so.element),
                    "to_f32": _TO_F32.get(so.element, ""),
                    "from_f32": _FROM_F32.get(so.element, ""),
                    "groups": so.groups,
                }
            )

        dst_dict = {
            "name": dst_name,
            "reg_field": dst_name,
            "element": dst_op.element,
            "c_type": dst_info.c_type,
            "width_bits": dst_info.width_bits,
            "is_float": dst_info.is_float,
            "needs_conv": _needs_conversion(dst_op.element),
            "to_f32": _TO_F32.get(dst_op.element, ""),
            "from_f32": _FROM_F32.get(dst_op.element, ""),
        }

        # Compute dot-product grouping ratio from first source
        group_size = 1
        if src_infos:
            first_src_info = get_type_info(sources[src_names[0]].element)
            group_size = _compute_ratio(first_src_info, dst_info)
            # If operand explicitly specifies groups, prefer that
            first_groups = sources[src_names[0]].groups
            if first_groups > 1:
                group_size = first_groups

        # Determine source SEW bits (for widening ops, vsetvli uses source width)
        src_sew_bits = 8  # default
        if src_infos:
            first_src_info = get_type_info(sources[src_names[0]].element)
            src_sew_bits = first_src_info.sew_bits

        # Determine if this is a widening operation (sources narrower than dest)
        is_widening = src_sew_bits < dst_info.sew_bits

        # Determine the SEW that Spike should require
        # For widening ops (DOT_PRODUCT, MAC with narrow sources), use source SEW
        if is_widening and op in (
            SemanticOp.DOT_PRODUCT,
            SemanticOp.MAC,
            SemanticOp.OUTER_PRODUCT,
        ):
            sew_bits = src_sew_bits
        else:
            sew_bits = dst_info.sew_bits

        # Determine the operation sub-template
        body_template_name = _OP_TEMPLATE.get(op, "execute_body/fma.cc.j2")

        return {
            "spec": spec,
            "op": op,
            "sources": src_infos,
            "dest": dst_dict,
            "group_size": group_size,
            "sew_bits": sew_bits,
            "src_sew_bits": src_sew_bits,
            "is_widening": is_widening,
            "accumulate": spec.semantics.accumulate,
            "saturation": spec.semantics.saturation,
            "body_template": body_template_name,
            "is_reduction": op in (SemanticOp.REDUCTION_SUM, SemanticOp.REDUCTION_MAX),
            "is_outer_product": op == SemanticOp.OUTER_PRODUCT,
            "is_fused_exp": op == SemanticOp.FUSED_EXP,
        }

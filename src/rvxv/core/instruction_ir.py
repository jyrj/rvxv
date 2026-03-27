"""Instruction IR — the heart of the RVXV system.

Pydantic v2 models that represent the complete specification of custom RISC-V
instructions, including encoding, operands, semantics, and constraints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from rvxv.core.type_system import ElementType, SemanticOp


class EncodingSpec(BaseModel):
    """Instruction binary encoding specification."""

    format: Literal["R-type", "R4-type", "I-type", "vector"]
    opcode: int = Field(ge=0, le=0x7F, description="7-bit opcode")
    funct3: int = Field(ge=0, le=0x7, description="3-bit function code")
    funct7: int | None = Field(
        default=None, ge=0, le=0x7F, description="7-bit function code (R-type)"
    )
    funct6: int | None = Field(
        default=None, ge=0, le=0x3F, description="6-bit function code (vector)"
    )
    funct2: int | None = Field(
        default=None, ge=0, le=0x3, description="2-bit function code (R4-type)"
    )
    vm: bool = Field(default=True, description="Vector masking enabled")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def match_value(self) -> int:
        """Compute the MATCH constant for instruction decoding.

        For R-type:
        [funct7(31:25) | rs2(24:20) | rs1(19:15) | funct3(14:12) | rd(11:7) | opcode(6:0)]
        match encodes funct7, funct3, opcode (rs2, rs1, rd are wildcards = 0)

        For vector:
        [funct6(31:26) | vm(25) | vs2(24:20) | vs1(19:15) | funct3(14:12) | vd(11:7) | opcode(6:0)]
        """
        value = self.opcode  # bits 6:0
        value |= (self.funct3 & 0x7) << 12  # bits 14:12

        if self.format == "R-type" and self.funct7 is not None:
            value |= (self.funct7 & 0x7F) << 25  # bits 31:25
        elif self.format == "vector" and self.funct6 is not None:
            value |= (self.funct6 & 0x3F) << 26  # bits 31:26
            if not self.vm:
                value |= 1 << 25  # vm bit
        elif self.format == "R4-type" and self.funct2 is not None:
            value |= (self.funct2 & 0x3) << 25  # bits 26:25

        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mask_value(self) -> int:
        """Compute the MASK constant -- which bits must match.

        A bit set in MASK means that bit position is significant for decoding.
        """
        mask = 0x7F  # opcode bits always significant
        mask |= 0x7 << 12  # funct3 always significant

        if self.format == "R-type" and self.funct7 is not None:
            mask |= 0x7F << 25  # funct7 significant
        elif self.format == "vector" and self.funct6 is not None:
            mask |= 0x3F << 26  # funct6 significant
            if not self.vm:
                mask |= 1 << 25  # vm bit significant
        elif self.format == "R4-type" and self.funct2 is not None:
            mask |= 0x3 << 25  # funct2 significant

        return mask

    @model_validator(mode="after")
    def validate_format_fields(self) -> EncodingSpec:
        if self.format == "R-type" and self.funct7 is None:
            raise ValueError("R-type format requires funct7")
        if self.format == "vector" and self.funct6 is None:
            raise ValueError("vector format requires funct6")
        if self.format == "R4-type" and self.funct2 is None:
            raise ValueError("R4-type format requires funct2")
        return self


class OperandSpec(BaseModel):
    """Specification of an instruction operand."""

    model_config = {"validate_assignment": True}

    type: Literal["scalar", "vector"]
    element: ElementType
    groups: int = Field(default=1, ge=1, description="Element group size for dot products")
    register_class: Literal["gpr", "fpr", "vr"] = "vr"

    @model_validator(mode="after")
    def validate_register_class(self) -> OperandSpec:
        if self.type == "scalar" and self.register_class == "vr":
            # Auto-assign register class for scalars based on element type
            from rvxv.core.type_system import get_type_info

            info = get_type_info(self.element)
            self.register_class = "fpr" if info.is_float else "gpr"
        return self


class SemanticsSpec(BaseModel):
    """Instruction semantic behavior specification."""

    operation: SemanticOp
    accumulate: bool = Field(default=False, description="Accumulate into destination")
    saturation: Literal["none", "signed", "unsigned"] = "none"
    rounding: Literal["rne", "rtz", "rdn", "rup", "rmm"] = "rne"


class ConstraintsSpec(BaseModel):
    """Instruction execution constraints."""

    vl_dependent: bool = Field(default=True, description="Operation respects vl")
    mask_agnostic: bool = Field(default=True, description="Masked elements are agnostic")
    tail_agnostic: bool = Field(default=True, description="Tail elements are agnostic")
    min_sew: int | None = Field(default=None, description="Minimum required SEW")
    max_sew: int | None = Field(default=None, description="Maximum allowed SEW")
    required_lmul: list[int] | None = Field(default=None, description="Allowed LMUL values")


class InstructionSpec(BaseModel):
    """Complete specification of a custom RISC-V instruction."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_.]*$", description="Instruction mnemonic")
    description: str = Field(min_length=1, description="Human-readable description")
    encoding: EncodingSpec
    operands: dict[str, OperandSpec] = Field(min_length=1)
    semantics: SemanticsSpec
    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)

    @model_validator(mode="after")
    def validate_operand_consistency(self) -> InstructionSpec:
        """Validate that operands are consistent with the semantic operation."""
        op = self.semantics.operation
        sources = [
            k for k in self.operands if k.startswith("vs") or k.startswith("rs")
        ]
        dest = [
            k for k in self.operands if k.startswith("vd") or k.startswith("rd")
        ]

        if op in (
            SemanticOp.DOT_PRODUCT,
            SemanticOp.FMA,
            SemanticOp.MULTIPLY,
            SemanticOp.OUTER_PRODUCT,
        ):
            if len(sources) < 2:
                raise ValueError(f"{op.value} requires at least 2 source operands")
        if not dest:
            raise ValueError("Instruction must have at least one destination operand")
        return self

    @property
    def source_operands(self) -> dict[str, OperandSpec]:
        return {
            k: v
            for k, v in self.operands.items()
            if k.startswith("vs") or k.startswith("rs")
        }

    @property
    def dest_operands(self) -> dict[str, OperandSpec]:
        return {
            k: v
            for k, v in self.operands.items()
            if k.startswith("vd") or k.startswith("rd")
        }

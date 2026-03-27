"""RISC-V assembly emitter that builds .S test files.

Produces assembly files compatible with the riscv-tests framework for
execution on Spike or standard RTL simulation testbenches.  Files follow
the RVTEST_CODE_BEGIN / RVTEST_CODE_END / RVTEST_DATA_BEGIN / RVTEST_DATA_END
conventions and use the tohost pass/fail mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from rvxv.core.type_system import ElementType, get_type_info


@dataclass
class DataEntry:
    """A single entry in the .data section."""

    label: str
    directive: str  # .byte, .half, .word, .dword, .space
    values: list[int | float]


class RISCVAssemblyEmitter:
    """Emits RISC-V assembly test programs (.S files).

    Follows the riscv-tests framework conventions for compatibility
    with Spike and standard RTL testbenches.
    """

    def __init__(self, test_name: str, description: str) -> None:
        self.test_name = test_name
        self.description = description
        self._code_lines: list[str] = []
        self._data_entries: list[DataEntry] = []
        self._test_num = 0
        self._temp_reg_counter = 0

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------

    def emit_header(self) -> None:
        """Emit standard riscv-tests preamble (RVTEST macros, etc.)."""
        self.emit_comment("Initialize test number")
        self.emit_li("TESTNUM", 0)

    def emit_footer(self) -> None:
        """Emit standard riscv-tests pass/fail reporting via tohost."""
        self.emit_blank()
        self.emit_comment("All tests passed")
        self.emit_raw("  TEST_PASSFAIL")
        self.emit_blank()
        self.emit_raw("RVTEST_CODE_END")

    # ------------------------------------------------------------------
    # Vector configuration
    # ------------------------------------------------------------------

    def emit_vsetvli(
        self,
        rd: str,
        rs1: str,
        sew: int,
        lmul: int = 1,
        ta: bool = True,
        ma: bool = True,
    ) -> None:
        """Emit a ``vsetvli`` instruction.

        Parameters
        ----------
        rd : str
            Destination register for the returned VL.
        rs1 : str
            Source register holding the requested VL (or ``zero`` for VLMAX).
        sew : int
            Element width in bits (8, 16, 32, 64).
        lmul : int
            LMUL value.  Positive values map to m1..m8, negative to mf2..mf8.
        ta, ma : bool
            Tail-agnostic and mask-agnostic policy flags.
        """
        sew_str = f"e{sew}"
        if lmul >= 1:
            lmul_str = f"m{lmul}"
        else:
            lmul_str = f"mf{int(1 / lmul)}"
        ta_str = "ta" if ta else "tu"
        ma_str = "ma" if ma else "mu"
        self._code_lines.append(
            f"  vsetvli {rd}, {rs1}, {sew_str}, {lmul_str}, {ta_str}, {ma_str}"
        )

    def emit_vsetivli(
        self,
        rd: str,
        avl: int,
        sew: int,
        lmul: int = 1,
        ta: bool = True,
        ma: bool = True,
    ) -> None:
        """Emit a ``vsetivli`` instruction with an immediate AVL."""
        sew_str = f"e{sew}"
        if lmul >= 1:
            lmul_str = f"m{lmul}"
        else:
            lmul_str = f"mf{int(1 / lmul)}"
        ta_str = "ta" if ta else "tu"
        ma_str = "ma" if ma else "mu"
        self._code_lines.append(
            f"  vsetivli {rd}, {avl}, {sew_str}, {lmul_str}, {ta_str}, {ma_str}"
        )

    # ------------------------------------------------------------------
    # Vector memory operations
    # ------------------------------------------------------------------

    def emit_vector_load(self, vd: str, base_reg: str, eew: int) -> None:
        """Emit a unit-stride vector load (``vleN.v``)."""
        self._code_lines.append(f"  vle{eew}.v {vd}, ({base_reg})")

    def emit_vector_store(self, vs: str, base_reg: str, eew: int) -> None:
        """Emit a unit-stride vector store (``vseN.v``)."""
        self._code_lines.append(f"  vse{eew}.v {vs}, ({base_reg})")

    # ------------------------------------------------------------------
    # Scalar memory helpers
    # ------------------------------------------------------------------

    def emit_load_word(self, rd: str, offset: int, base: str) -> None:
        """Emit ``lw rd, offset(base)``."""
        self._code_lines.append(f"  lw {rd}, {offset}({base})")

    def emit_store_word(self, rs: str, offset: int, base: str) -> None:
        """Emit ``sw rs, offset(base)``."""
        self._code_lines.append(f"  sw {rs}, {offset}({base})")

    def emit_load_dword(self, rd: str, offset: int, base: str) -> None:
        """Emit ``ld rd, offset(base)``."""
        self._code_lines.append(f"  ld {rd}, {offset}({base})")

    def emit_store_dword(self, rs: str, offset: int, base: str) -> None:
        """Emit ``sd rs, offset(base)``."""
        self._code_lines.append(f"  sd {rs}, {offset}({base})")

    # ------------------------------------------------------------------
    # Custom (encoded) instructions
    # ------------------------------------------------------------------

    # RISC-V register name -> encoding number mapping
    _REG_NUM: dict[str, int] = {
        "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
        "t0": 5, "t1": 6, "t2": 7,
        "s0": 8, "fp": 8, "s1": 9,
        "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
        "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23,
        "s8": 24, "s9": 25, "s10": 26, "s11": 27,
        "t3": 28, "t4": 29, "t5": 30, "t6": 31,
        # Vector registers v0..v31
        **{f"v{i}": i for i in range(32)},
        # Floating-point registers
        **{f"f{i}": i for i in range(32)},
        **{f"ft{i}": i for i in range(8)},
        "fs0": 8, "fs1": 9,
        "fa0": 10, "fa1": 11, "fa2": 12, "fa3": 13, "fa4": 14, "fa5": 15, "fa6": 16, "fa7": 17,
        "fs2": 18, "fs3": 19, "fs4": 20, "fs5": 21, "fs6": 22, "fs7": 23,
        "fs8": 24, "fs9": 25, "fs10": 26, "fs11": 27,
        "ft8": 28, "ft9": 29, "ft10": 30, "ft11": 31,
    }

    @classmethod
    def _reg_num(cls, name: str) -> int:
        """Return the 5-bit register number for a RISC-V register name."""
        return cls._REG_NUM.get(name, 0)

    @staticmethod
    def _encode_r_type_word(
        opcode: int,
        funct3: int,
        funct7: int,
        rd: str,
        rs1: str,
        rs2: str,
    ) -> int:
        """Compute a 32-bit R-type instruction word from fields."""
        rd_n = RISCVAssemblyEmitter._reg_num(rd) & 0x1F
        rs1_n = RISCVAssemblyEmitter._reg_num(rs1) & 0x1F
        rs2_n = RISCVAssemblyEmitter._reg_num(rs2) & 0x1F
        word = (opcode & 0x7F)
        word |= (rd_n << 7)
        word |= ((funct3 & 0x7) << 12)
        word |= (rs1_n << 15)
        word |= (rs2_n << 20)
        word |= ((funct7 & 0x7F) << 25)
        return word

    def emit_custom_insn(
        self,
        fmt: str,
        opcode: int,
        funct3: int,
        funct7: int | None,
        rd: str,
        rs1: str,
        rs2: str,
    ) -> None:
        """Emit a custom instruction via the ``.insn`` assembler directive.

        Currently supports R-type and vector (OPFVV-style) formats.
        Also appends a comment showing the equivalent .word encoding.
        """
        if fmt in ("R-type", "vector"):
            f7 = funct7 if funct7 is not None else 0
            word = self._encode_r_type_word(opcode, funct3, f7, rd, rs1, rs2)
            # Use .word encoding for maximum assembler compatibility.
            # GCC's .insn r does not support vector register operands
            # with large funct7 values.
            self._code_lines.append(
                f"  .word 0x{word:08x}"
                f"  # {rd} <- dot({rs2}, {rs1})"
            )
        elif fmt == "R4-type":
            # R4-type uses .insn r4
            f2 = funct7 if funct7 is not None else 0  # re-using funct7 slot for funct2
            self._code_lines.append(
                f"  .insn r4 0x{opcode:02x}, 0x{funct3:x}, 0x{f2:02x}, "
                f"{rd}, {rs1}, {rs2}, {rd}"
            )
        elif fmt == "I-type":
            # I-type: .insn i opcode, funct3, rd, rs1, imm
            imm = funct7 if funct7 is not None else 0
            self._code_lines.append(
                f"  .insn i 0x{opcode:02x}, 0x{funct3:x}, {rd}, {rs1}, {imm}"
            )

    def emit_custom_insn_word(
        self,
        opcode: int,
        funct3: int,
        funct7: int | None,
        rd: str,
        rs1: str,
        rs2: str,
    ) -> None:
        """Emit a custom instruction as a raw ``.word`` 32-bit encoding.

        This provides maximum assembler compatibility by encoding the full
        instruction word directly, bypassing ``.insn`` which may not be
        supported by all assembler versions.
        """
        f7 = funct7 if funct7 is not None else 0
        word = self._encode_r_type_word(opcode, funct3, f7, rd, rs1, rs2)
        self._code_lines.append(
            f"  .word 0x{word:08x}  # {rd} = op({rs1}, {rs2})"
        )

    def emit_custom_insn_raw(self, word: int) -> None:
        """Emit a raw 32-bit instruction word via ``.word``."""
        self._code_lines.append(f"  .word 0x{word:08x}")

    # ------------------------------------------------------------------
    # Scalar ALU / address helpers
    # ------------------------------------------------------------------

    def emit_load_address(self, reg: str, label: str) -> None:
        """Load the address of *label* into *reg* (``la`` pseudo-instruction)."""
        self._code_lines.append(f"  la {reg}, {label}")

    def emit_li(self, reg: str, value: int) -> None:
        """Load an immediate integer value into *reg*."""
        self._code_lines.append(f"  li {reg}, {value}")

    def emit_mv(self, rd: str, rs: str) -> None:
        """Register move (``mv rd, rs``)."""
        self._code_lines.append(f"  mv {rd}, {rs}")

    def emit_add(self, rd: str, rs1: str, rs2: str) -> None:
        """Emit ``add rd, rs1, rs2``."""
        self._code_lines.append(f"  add {rd}, {rs1}, {rs2}")

    def emit_addi(self, rd: str, rs1: str, imm: int) -> None:
        """Emit ``addi rd, rs1, imm``."""
        self._code_lines.append(f"  addi {rd}, {rs1}, {imm}")

    def emit_slli(self, rd: str, rs1: str, shamt: int) -> None:
        """Emit ``slli rd, rs1, shamt``."""
        self._code_lines.append(f"  slli {rd}, {rs1}, {shamt}")

    # ------------------------------------------------------------------
    # VMV helpers
    # ------------------------------------------------------------------

    def emit_vmv_v_x(self, vd: str, rs: str) -> None:
        """Emit ``vmv.v.x vd, rs`` -- splat scalar to vector."""
        self._code_lines.append(f"  vmv.v.x {vd}, {rs}")

    def emit_vmv_v_i(self, vd: str, imm: int) -> None:
        """Emit ``vmv.v.i vd, imm`` -- splat immediate to vector."""
        self._code_lines.append(f"  vmv.v.i {vd}, {imm}")

    def emit_vmv_x_s(self, rd: str, vs: str) -> None:
        """Emit ``vmv.x.s rd, vs`` -- extract element 0 to scalar."""
        self._code_lines.append(f"  vmv.x.s {rd}, {vs}")

    # ------------------------------------------------------------------
    # CSR helpers
    # ------------------------------------------------------------------

    def emit_csrwi(self, csr: str, imm: int) -> None:
        """Emit ``csrwi csr, imm`` (write immediate to CSR)."""
        self._code_lines.append(f"  csrwi {csr}, {imm}")

    def emit_csrw(self, csr: str, rs: str) -> None:
        """Emit ``csrw csr, rs`` (write register to CSR)."""
        self._code_lines.append(f"  csrw {csr}, {rs}")

    # ------------------------------------------------------------------
    # Branch / comparison helpers
    # ------------------------------------------------------------------

    def emit_branch(self, op: str, rs1: str, rs2: str, label: str) -> None:
        """Emit a conditional branch (``beq``, ``bne``, etc.)."""
        self._code_lines.append(f"  {op} {rs1}, {rs2}, {label}")

    def emit_pass_fail_check(
        self, result_reg: str, expected_reg: str, test_id: int
    ) -> None:
        """Emit a comparison that branches to *fail* on mismatch.

        Also updates ``TESTNUM`` so that Spike reports which test failed.
        """
        self._code_lines.append(f"  li TESTNUM, {test_id}")
        self._code_lines.append(f"  bne {result_reg}, {expected_reg}, fail")

    def emit_jump(self, label: str) -> None:
        """Emit unconditional jump (``j label``)."""
        self._code_lines.append(f"  j {label}")

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def emit_comment(self, text: str) -> None:
        """Emit an assembly comment line."""
        self._code_lines.append(f"  # {text}")

    def emit_label(self, name: str) -> None:
        """Emit a label on its own line."""
        self._code_lines.append(f"{name}:")

    def emit_blank(self) -> None:
        """Emit an empty line for readability."""
        self._code_lines.append("")

    def emit_raw(self, line: str) -> None:
        """Emit a raw assembly line (no processing)."""
        self._code_lines.append(line)

    # ------------------------------------------------------------------
    # Test case structure helpers
    # ------------------------------------------------------------------

    def begin_test_case(self, test_id: int, description: str) -> None:
        """Begin a numbered test case with a comment header."""
        self._test_num = test_id
        self.emit_blank()
        self.emit_comment(f"Test {test_id}: {description}")
        self.emit_raw(f"test_{test_id}:")

    def end_test_case(self) -> None:
        """Optional separator after a test case."""
        self.emit_blank()

    # ------------------------------------------------------------------
    # Data section helpers
    # ------------------------------------------------------------------

    def add_test_data(
        self, label: str, element_type: ElementType, values: list[int]
    ) -> None:
        """Add typed test data to the ``.data`` section.

        The appropriate assembler directive (``.byte``, ``.half``, ``.word``,
        ``.dword``) is selected based on the element type width.
        """
        info = get_type_info(element_type)
        if info.width_bits <= 8:
            directive = ".byte"
        elif info.width_bits <= 16:
            directive = ".half"
        elif info.width_bits <= 32:
            directive = ".word"
        else:
            directive = ".dword"
        self._data_entries.append(DataEntry(label, directive, values))

    def add_raw_data(self, label: str, directive: str, values: list[int]) -> None:
        """Add raw data with an explicit directive."""
        self._data_entries.append(DataEntry(label, directive, values))

    def add_result_space(self, label: str, num_bytes: int) -> None:
        """Reserve zeroed space for results in the ``.data`` section."""
        self._data_entries.append(DataEntry(label, ".space", [num_bytes]))

    # ------------------------------------------------------------------
    # Temporary register allocation
    # ------------------------------------------------------------------

    def alloc_temp_gpr(self) -> str:
        """Return the name of the next available temporary GPR (t0-t6)."""
        regs = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]
        if self._temp_reg_counter >= len(regs):
            raise RuntimeError("Exhausted temporary GPRs")
        reg = regs[self._temp_reg_counter]
        self._temp_reg_counter += 1
        return reg

    def reset_temp_gprs(self) -> None:
        """Reset the temporary GPR allocator (call between test cases)."""
        self._temp_reg_counter = 0

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the complete assembly file as a string."""
        lines: list[str] = []

        # ---- File header ----
        lines.append(
            "# Auto-generated by RVXV "
            "\u2014 RISC-V AI Extension Verification Platform"
        )
        lines.append(f"# Test: {self.test_name}")
        lines.append(f"# Description: {self.description}")
        lines.append("# Do not edit manually. Regenerate with: rvxv generate")
        lines.append("")
        lines.append('#include "riscv_test.h"')
        lines.append('#include "test_macros.h"')
        lines.append("")
        lines.append("RVTEST_RV64UV")
        lines.append("RVTEST_CODE_BEGIN")
        lines.append("")

        # ---- Code section ----
        lines.extend(self._code_lines)

        # ---- Pass / fail epilogue ----
        lines.append("")
        lines.append("  # All tests passed")
        lines.append("  TEST_PASSFAIL")
        lines.append("")
        lines.append("RVTEST_CODE_END")
        lines.append("")

        # ---- Data section ----
        lines.append("  .data")
        lines.append("RVTEST_DATA_BEGIN")
        lines.append("")

        for entry in self._data_entries:
            lines.append("  .balign 16")
            lines.append(f"{entry.label}:")
            if entry.directive == ".space":
                lines.append(f"  .space {entry.values[0]}")
            else:
                vals = entry.values
                for i in range(0, len(vals), 8):
                    chunk = vals[i : i + 8]
                    val_str = ", ".join(str(v) for v in chunk)
                    lines.append(f"  {entry.directive} {val_str}")
            lines.append("")

        lines.append("RVTEST_DATA_END")
        lines.append("")

        return "\n".join(lines)

from __future__ import annotations


class Int4Signed:
    """4-bit signed integer: range [-8, +7]."""
    MIN = -8
    MAX = 7
    BITS = 4

    @staticmethod
    def encode(value: int) -> int:
        """Encode integer to 4-bit signed representation (two's complement)."""
        if value < Int4Signed.MIN or value > Int4Signed.MAX:
            raise ValueError(
                f"Value {value} out of INT4 signed range"
                f" [{Int4Signed.MIN}, {Int4Signed.MAX}]"
            )
        return value & 0xF

    @staticmethod
    def decode(bits: int) -> int:
        """Decode 4-bit two's complement to integer."""
        bits &= 0xF
        if bits & 0x8:  # negative
            return bits - 16
        return bits

    @staticmethod
    def saturate(value: int) -> int:
        """Clamp value to INT4 signed range."""
        return max(Int4Signed.MIN, min(Int4Signed.MAX, value))

    @staticmethod
    def pack_8(values: list[int]) -> int:
        """Pack 8 INT4 values into a 32-bit word."""
        if len(values) != 8:
            raise ValueError("Must provide exactly 8 values for packing")
        result = 0
        for i, v in enumerate(values):
            result |= (Int4Signed.encode(v) & 0xF) << (i * 4)
        return result

    @staticmethod
    def unpack_8(packed: int) -> list[int]:
        """Unpack a 32-bit word into 8 INT4 signed values."""
        return [Int4Signed.decode((packed >> (i * 4)) & 0xF) for i in range(8)]


class Int4Unsigned:
    """4-bit unsigned integer: range [0, 15]."""
    MIN = 0
    MAX = 15
    BITS = 4

    @staticmethod
    def encode(value: int) -> int:
        if value < 0 or value > 15:
            raise ValueError(f"Value {value} out of UINT4 range [0, 15]")
        return value & 0xF

    @staticmethod
    def decode(bits: int) -> int:
        return bits & 0xF

    @staticmethod
    def saturate(value: int) -> int:
        return max(0, min(15, value))

    @staticmethod
    def pack_8(values: list[int]) -> int:
        if len(values) != 8:
            raise ValueError("Must provide exactly 8 values for packing")
        result = 0
        for i, v in enumerate(values):
            result |= (Int4Unsigned.encode(v) & 0xF) << (i * 4)
        return result

    @staticmethod
    def unpack_8(packed: int) -> list[int]:
        return [Int4Unsigned.decode((packed >> (i * 4)) & 0xF) for i in range(8)]

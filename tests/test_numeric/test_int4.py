"""Tests for INT4 signed/unsigned types."""

from __future__ import annotations

import pytest

from rvxv.numeric.int4 import Int4Signed, Int4Unsigned


class TestInt4Signed:
    def test_range(self):
        assert Int4Signed.MIN == -8
        assert Int4Signed.MAX == 7

    def test_encode_decode_roundtrip(self):
        for val in range(-8, 8):
            bits = Int4Signed.encode(val)
            decoded = Int4Signed.decode(bits)
            assert decoded == val, f"Roundtrip failed for {val}"

    def test_encode_out_of_range(self):
        with pytest.raises(ValueError):
            Int4Signed.encode(8)
        with pytest.raises(ValueError):
            Int4Signed.encode(-9)

    def test_saturate(self):
        assert Int4Signed.saturate(10) == 7
        assert Int4Signed.saturate(-10) == -8
        assert Int4Signed.saturate(0) == 0
        assert Int4Signed.saturate(7) == 7
        assert Int4Signed.saturate(-8) == -8

    def test_pack_unpack(self):
        values = [-8, -1, 0, 1, 7, -3, 4, -5]
        packed = Int4Signed.pack_8(values)
        unpacked = Int4Signed.unpack_8(packed)
        assert unpacked == values

    def test_twos_complement(self):
        assert Int4Signed.encode(-1) == 0xF  # 1111
        assert Int4Signed.encode(-8) == 0x8  # 1000
        assert Int4Signed.encode(7) == 0x7   # 0111

    def test_pack_wrong_count(self):
        with pytest.raises(ValueError):
            Int4Signed.pack_8([1, 2, 3])


class TestInt4Unsigned:
    def test_range(self):
        assert Int4Unsigned.MIN == 0
        assert Int4Unsigned.MAX == 15

    def test_encode_decode_roundtrip(self):
        for val in range(16):
            bits = Int4Unsigned.encode(val)
            decoded = Int4Unsigned.decode(bits)
            assert decoded == val

    def test_encode_out_of_range(self):
        with pytest.raises(ValueError):
            Int4Unsigned.encode(16)
        with pytest.raises(ValueError):
            Int4Unsigned.encode(-1)

    def test_saturate(self):
        assert Int4Unsigned.saturate(20) == 15
        assert Int4Unsigned.saturate(-5) == 0
        assert Int4Unsigned.saturate(8) == 8

    def test_pack_unpack(self):
        values = [0, 1, 2, 3, 15, 14, 13, 12]
        packed = Int4Unsigned.pack_8(values)
        unpacked = Int4Unsigned.unpack_8(packed)
        assert unpacked == values

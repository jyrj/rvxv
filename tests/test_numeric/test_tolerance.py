"""Tests for ULP distance and tolerance comparison."""

from __future__ import annotations

import math

import pytest

from rvxv.numeric.tolerance import (
    TOLERANCE_EXACT,
    TOLERANCE_FMA,
    ToleranceSpec,
    compare_with_tolerance,
    ulp_distance,
    ulp_of_value,
)


class TestULPDistance:
    def test_same_value(self):
        assert ulp_distance(1.0, 1.0) == 0

    def test_adjacent_floats(self):
        import struct
        bits = struct.unpack(">I", struct.pack(">f", 1.0))[0]
        next_val = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
        assert ulp_distance(1.0, next_val) == 1

    def test_nan_returns_negative(self):
        assert ulp_distance(float("nan"), 1.0) == -1
        assert ulp_distance(1.0, float("nan")) == -1

    def test_symmetry(self):
        assert ulp_distance(1.0, 2.0) == ulp_distance(2.0, 1.0)

    def test_zero_to_min_subnormal(self):
        dist = ulp_distance(0.0, 1.4e-45)  # Smallest FP32 subnormal
        assert dist >= 0


class TestULPOfValue:
    def test_ulp_of_one(self):
        ulp = ulp_of_value(1.0)
        assert ulp == pytest.approx(1.1920929e-7, rel=1e-5)

    def test_ulp_of_nan(self):
        assert math.isnan(ulp_of_value(float("nan")))

    def test_ulp_of_inf(self):
        assert math.isnan(ulp_of_value(float("inf")))


class TestCompareWithTolerance:
    def test_exact_match(self):
        result = compare_with_tolerance(1.0, 1.0, TOLERANCE_EXACT)
        assert result.passed is True
        assert result.ulp_dist == 0

    def test_exact_mismatch(self):
        result = compare_with_tolerance(1.0, 1.0 + 1e-7, TOLERANCE_EXACT)
        assert result.passed is False

    def test_fma_tolerance(self):
        import struct
        bits = struct.unpack(">I", struct.pack(">f", 1.0))[0]
        next_val = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
        result = compare_with_tolerance(next_val, 1.0, TOLERANCE_FMA)
        assert result.passed is True
        assert result.ulp_dist == 1

    def test_nan_equal(self):
        result = compare_with_tolerance(float("nan"), float("nan"),
                                        ToleranceSpec(nan_equal=True))
        assert result.passed is True

    def test_nan_not_equal(self):
        result = compare_with_tolerance(float("nan"), float("nan"),
                                        ToleranceSpec(nan_equal=False))
        assert result.passed is False

    def test_infinity_match(self):
        result = compare_with_tolerance(float("inf"), float("inf"), TOLERANCE_EXACT)
        assert result.passed is True

    def test_infinity_sign_mismatch(self):
        result = compare_with_tolerance(float("inf"), float("-inf"), TOLERANCE_EXACT)
        assert result.passed is False

    def test_zero_sign_matters(self):
        result = compare_with_tolerance(-0.0, 0.0,
                                        ToleranceSpec(zero_sign_matters=True))
        assert result.passed is False

    def test_zero_sign_ignored(self):
        result = compare_with_tolerance(-0.0, 0.0,
                                        ToleranceSpec(zero_sign_matters=False))
        assert result.passed is True

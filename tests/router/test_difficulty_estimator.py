"""Tests for difficulty_estimator -- signal fusion."""
import os
import pytest
from models.route import SignalBundle, DifficultyBin
from router.difficulty_estimator import (
    estimate_difficulty,
    get_difficulty_bin,
    WEIGHTS_3_SIGNAL,
    WEIGHTS_4_SIGNAL,
)


class TestEstimateDifficulty:
    def test_all_zero_3sig(self):
        s = SignalBundle()
        d = estimate_difficulty(s, WEIGHTS_3_SIGNAL)
        # (1-0)*0.4 + (1-0)*0.3 + 0*0.3 + 0*0.0 = 0.7
        assert d == pytest.approx(0.7, abs=0.01)

    def test_all_one_3sig(self):
        s = SignalBundle(
            pattern_cache_score=1.0, retrieval_confidence=1.0,
            query_complexity=1.0, geometric_energy=1.0,
        )
        d = estimate_difficulty(s, WEIGHTS_3_SIGNAL)
        # (1-1)*0.4 + (1-1)*0.3 + 1*0.3 + 1*0.0 = 0.3
        assert d == pytest.approx(0.3, abs=0.01)

    def test_all_zero_4sig(self):
        s = SignalBundle()
        d = estimate_difficulty(s, WEIGHTS_4_SIGNAL)
        # (1-0)*0.30 + (1-0)*0.25 + 0*0.20 + 0*0.25 = 0.55
        assert d == pytest.approx(0.55, abs=0.01)

    def test_all_one_4sig(self):
        s = SignalBundle(
            pattern_cache_score=1.0, retrieval_confidence=1.0,
            query_complexity=1.0, geometric_energy=1.0,
        )
        d = estimate_difficulty(s, WEIGHTS_4_SIGNAL)
        # 0*0.30 + 0*0.25 + 1*0.20 + 1*0.25 = 0.45
        assert d == pytest.approx(0.45, abs=0.01)

    def test_high_cache_lowers_difficulty(self):
        low = SignalBundle(pattern_cache_score=0.0)
        high = SignalBundle(pattern_cache_score=1.0)
        assert estimate_difficulty(high) < estimate_difficulty(low)

    def test_high_complexity_raises_difficulty(self):
        low = SignalBundle(query_complexity=0.0)
        high = SignalBundle(query_complexity=1.0)
        assert estimate_difficulty(high) > estimate_difficulty(low)

    def test_clamped_to_unit(self):
        s = SignalBundle()
        d = estimate_difficulty(s)
        assert 0.0 <= d <= 1.0

    def test_custom_weights(self):
        s = SignalBundle(pattern_cache_score=0.5, query_complexity=0.5)
        w = {"pattern_cache": 1.0, "retrieval_confidence": 0.0,
             "query_complexity": 0.0, "geometric_energy": 0.0}
        d = estimate_difficulty(s, w)
        assert d == pytest.approx(0.5, abs=0.01)  # (1-0.5)*1.0

    def test_env_selects_4sig(self, monkeypatch):
        monkeypatch.setenv("GEOMETRIC_LENS_ENABLED", "true")
        s = SignalBundle(geometric_energy=1.0)
        d = estimate_difficulty(s)
        # geometric component contributes when 4-sig active
        s2 = SignalBundle(geometric_energy=0.0)
        d2 = estimate_difficulty(s2)
        assert d > d2

    def test_env_selects_3sig(self, monkeypatch):
        monkeypatch.setenv("GEOMETRIC_LENS_ENABLED", "false")
        s1 = SignalBundle(geometric_energy=1.0)
        s2 = SignalBundle(geometric_energy=0.0)
        # geometric has 0 weight in 3-sig, so no difference
        assert estimate_difficulty(s1) == pytest.approx(
            estimate_difficulty(s2), abs=0.001
        )


class TestGetDifficultyBin:
    def test_easy(self):
        assert get_difficulty_bin(0.0) == DifficultyBin.EASY
        assert get_difficulty_bin(0.29) == DifficultyBin.EASY

    def test_medium(self):
        assert get_difficulty_bin(0.3) == DifficultyBin.MEDIUM
        assert get_difficulty_bin(0.59) == DifficultyBin.MEDIUM

    def test_hard(self):
        assert get_difficulty_bin(0.6) == DifficultyBin.HARD
        assert get_difficulty_bin(1.0) == DifficultyBin.HARD

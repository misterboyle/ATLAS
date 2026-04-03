"""Tests for feedback_recorder -- outcome recording and stats."""
import pytest
from models.route import Route, DifficultyBin
from router.feedback_recorder import (
    record_outcome,
    get_routing_stats,
    reset_stats,
    KEY_PREFIX,
    STATS_PREFIX,
)


class TestRecordOutcome:
    def test_success_increments_alpha(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.STANDARD, True)
        k = f"{KEY_PREFIX}:easy:standard:alpha"
        assert float(mock_redis._store.get(k, 0)) == 1.0

    def test_failure_increments_beta(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.STANDARD, False)
        k = f"{KEY_PREFIX}:easy:standard:beta"
        assert float(mock_redis._store.get(k, 0)) == 1.0

    def test_success_does_not_touch_beta(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.FAST_PATH, True)
        k = f"{KEY_PREFIX}:easy:fast_path:beta"
        assert mock_redis._store.get(k) is None

    def test_multiple_outcomes_accumulate(self, mock_redis):
        for _ in range(5):
            record_outcome(mock_redis, DifficultyBin.HARD, Route.HARD_PATH, True)
        k = f"{KEY_PREFIX}:hard:hard_path:alpha"
        assert float(mock_redis._store[k]) == 5.0

    def test_stats_counters_updated(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.MEDIUM, Route.STANDARD, True)
        assert mock_redis._store[f"{STATS_PREFIX}:total_decisions"] == 1
        assert mock_redis._store[f"{STATS_PREFIX}:total_successes"] == 1
        assert mock_redis._store[f"{STATS_PREFIX}:route:standard"] == 1
        assert mock_redis._store[f"{STATS_PREFIX}:bin:medium"] == 1

    def test_failure_no_success_counter(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.FAST_PATH, False)
        assert mock_redis._store.get(f"{STATS_PREFIX}:total_successes") is None
        assert mock_redis._store[f"{STATS_PREFIX}:total_decisions"] == 1


class TestGetRoutingStats:
    def test_empty_stats(self, mock_redis):
        st = get_routing_stats(mock_redis)
        assert st["total_decisions"] == 0
        assert st["success_rate"] == 0.0

    def test_populated_stats(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.STANDARD, True)
        record_outcome(mock_redis, DifficultyBin.EASY, Route.STANDARD, False)
        st = get_routing_stats(mock_redis)
        assert st["total_decisions"] == 2
        assert st["total_successes"] == 1
        assert st["success_rate"] == 0.5
        assert st["route_distribution"]["standard"] == 2
        assert st["difficulty_distribution"]["easy"] == 2


class TestResetStats:
    def test_clears_all(self, mock_redis):
        record_outcome(mock_redis, DifficultyBin.EASY, Route.STANDARD, True)
        reset_stats(mock_redis)
        st = get_routing_stats(mock_redis)
        assert st["total_decisions"] == 0
        assert st["total_successes"] == 0

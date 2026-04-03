"""Tests for v3_dashboard_push -- native dashboard integration.

All tests use mocked Redis so no infrastructure is required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmark.v3_dashboard_push import push_task, push_summary


def _make_task_result(passed=True, task_id="LCB_1234", tokens=500,
                      time_ms=12000, candidates=3, phase="phase1"):
    return {
        "task_id": task_id,
        "passed": passed,
        "total_tokens": tokens,
        "total_time_ms": time_ms,
        "candidates_generated": candidates,
        "phase_solved": phase,
    }


def _mock_redis():
    r = MagicMock()
    pipe = MagicMock()
    r.pipeline.return_value = pipe
    return r, pipe


class TestPushTask:
    """push_task writes per-task metrics to Redis."""

    def test_pass_increments_passed_counter(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(passed=True), 1, 10, 5.0)
        calls = [str(c) for c in pipe.hincrby.call_args_list]
        assert any("'passed', 1" in c for c in calls)
        assert not any("'failed'" in c for c in calls)

    def test_fail_increments_failed_counter(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(passed=False), 1, 10, 5.0)
        calls = [str(c) for c in pipe.hincrby.call_args_list]
        assert any("'failed', 1" in c for c in calls)
        assert not any("'passed', 1" in c for c in calls)

    def test_recent_tasks_entry_has_correct_format(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(task_id="LCB_42"), 3, 10, 6.0)
        args = pipe.lpush.call_args[0]
        assert args[0] == "atlas:metrics:recent_tasks"
        entry = json.loads(args[1])
        assert entry["task_id"] == "LCB_42"
        assert entry["type"] == "v3_benchmark"
        assert entry["success"] is True
        assert entry["progress"] == "3/10"
        assert entry["phase"] == "phase1"
        assert entry["model"] == "qwen3-14b"
        assert "completed_at" in entry

    def test_uses_atlas_metrics_prefix(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(), 1, 1, 1.0)
        for call_obj in pipe.hincrby.call_args_list:
            assert call_obj.args[0].startswith("atlas:metrics:daily:")

    def test_uses_redis_pipeline(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(), 1, 1, 1.0)
        r.pipeline.assert_called_once()
        pipe.execute.assert_called_once()

    def test_trims_recent_tasks_to_50(self):
        r, pipe = _mock_redis()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(), 1, 1, 1.0)
        pipe.ltrim.assert_called_once_with(
            "atlas:metrics:recent_tasks", 0, 49
        )

    def test_silently_ignores_redis_unavailable(self):
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=None):
            push_task(_make_task_result(), 1, 1, 1.0)

    def test_silently_ignores_redis_error(self):
        r = MagicMock()
        r.pipeline.side_effect = ConnectionError("down")
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_task(_make_task_result(), 1, 1, 1.0)


class TestPushSummary:
    """push_summary writes final benchmark summary to Redis."""

    def test_pushes_run_complete_entry(self):
        r = MagicMock()
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_summary(7, 10, 0.7, {"phase1": 5, "pr_cot": 2})
        args = r.lpush.call_args[0]
        assert args[0] == "atlas:metrics:recent_tasks"
        entry = json.loads(args[1])
        assert entry["task_id"] == "RUN_COMPLETE"
        assert "7/10" in entry["progress"]
        assert "70.0%" in entry["progress"]
        assert entry["breakdown"] == {"phase1": 5, "pr_cot": 2}

    def test_silently_ignores_redis_unavailable(self):
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=None):
            push_summary(5, 10, 0.5, {})

    def test_silently_ignores_redis_error(self):
        r = MagicMock()
        r.lpush.side_effect = ConnectionError("down")
        with patch("benchmark.v3_dashboard_push._get_redis", return_value=r):
            push_summary(5, 10, 0.5, {})

"""Native dashboard integration for v3_runner.

Pushes per-task results and run summaries directly to Redis
so the ATLAS Dashboard updates in real-time without needing
the separate v3_dashboard_bridge.py process.

All functions are fail-safe -- they never raise exceptions.
If Redis is unreachable, they silently return.
"""

import json
import os
from datetime import datetime, timezone

_redis_conn = None
_redis_init_done = False


def _get_redis():
    """Lazy Redis connection. Returns None if unavailable."""
    global _redis_conn, _redis_init_done
    if _redis_init_done:
        return _redis_conn
    _redis_init_done = True
    try:
        import redis
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()
        _redis_conn = r
    except Exception:
        _redis_conn = None
    return _redis_conn


def push_task(task_result, done, total, rate):
    """Push a completed task result to dashboard Redis keys. Never raises."""
    try:
        r = _get_redis()
        if r is None:
            return
        today = datetime.now(timezone.utc).date().isoformat()
        daily_key = f"atlas:metrics:daily:{today}"

        tokens = task_result.get("total_tokens", 0)
        duration_ms = int(task_result.get("total_time_ms", 0))

        pipe = r.pipeline()
        pipe.hincrby(daily_key, "total", 1)
        pipe.hincrby(daily_key, "requests_total", 1)
        pipe.hincrby(daily_key, "total_attempts", 1)
        pipe.hincrby(daily_key, "total_tokens", tokens)
        pipe.hincrby(daily_key, "tokens_total", tokens)
        pipe.hincrby(daily_key, "total_duration_ms", duration_ms)
        if task_result.get("passed"):
            pipe.hincrby(daily_key, "passed", 1)
        else:
            pipe.hincrby(daily_key, "failed", 1)

        entry = json.dumps({
            "task_id": task_result.get("task_id", "unknown"),
            "type": "v3_benchmark",
            "success": bool(task_result.get("passed")),
            "attempts": task_result.get("candidates_generated", 1),
            "duration_ms": duration_ms,
            "tokens": tokens,
            "model": "qwen3-14b",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "progress": f"{done}/{total}",
            "rate_tasks_hr": int(rate),
            "phase": task_result.get("phase_solved", "unknown"),
        })
        pipe.lpush("atlas:metrics:recent_tasks", entry)
        pipe.ltrim("atlas:metrics:recent_tasks", 0, 49)
        pipe.execute()
    except Exception:
        pass  # Never crash benchmark for dashboard


def push_summary(passed, total, rate, breakdown):
    """Push final benchmark summary to dashboard. Never raises."""
    try:
        r = _get_redis()
        if r is None:
            return
        entry = json.dumps({
            "task_id": "RUN_COMPLETE",
            "type": "v3_benchmark (COMPLETE)",
            "success": True,
            "attempts": total,
            "duration_ms": 0,
            "tokens": 0,
            "model": "qwen3-14b",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "progress": f"{passed}/{total} ({rate*100:.1f}%)",
            "rate_tasks_hr": 0,
            "phase": "complete",
            "summary": f"BENCHMARK COMPLETE: {passed}/{total} ({rate*100:.1f}%)",
            "breakdown": breakdown,
        })
        r.lpush("atlas:metrics:recent_tasks", entry)
        r.ltrim("atlas:metrics:recent_tasks", 0, 49)
    except Exception:
        pass

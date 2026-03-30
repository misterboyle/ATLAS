"""Bridge V3 benchmark log output to ATLAS Dashboard via Redis.

Usage:
    python3 v3_dashboard_bridge.py /tmp/v3_phase3_stress.log

Reads the V3 runner log, parses task results, and pushes them to Redis
so the ATLAS Dashboard shows benchmark progress in real-time.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

try:
    import redis
except ImportError:
    print("pip install redis", file=sys.stderr)
    sys.exit(1)

TASK_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+):\s+(PASS|FAIL)"
    r"\s+\(via\s+(\S+),\s+(\d+)\s+tok\)"
    r"\s+\[(\d+)\s+tasks/hr\]"
)

COMPLETE_RE = re.compile(
    r"pass@1:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)"
)


def parse_line(line):
    m = TASK_RE.search(line)
    if not m:
        return None
    return {
        "idx": int(m.group(1)),
        "total": int(m.group(2)),
        "task_id": m.group(3),
        "result": m.group(4),
        "phase": m.group(5),
        "tokens": int(m.group(6)),
        "rate": int(m.group(7)),
    }


def check_completion(line, r):
    """Detect 'V3 BENCHMARK COMPLETE' summary and push to dashboard."""
    m = COMPLETE_RE.search(line)
    if not m:
        return False
    passed, total, rate = int(m.group(1)), int(m.group(2)), m.group(3)
    entry = json.dumps({
        "task_id": "RUN_COMPLETE",
        "type": f"v3_phase3_stress (COMPLETE)",
        "success": True,
        "attempts": total,
        "duration_ms": 0,
        "tokens": 0,
        "model": "qwen3:14b",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "progress": f"{passed}/{total} ({rate}%)",
        "rate_tasks_hr": 0,
        "summary": f"BENCHMARK COMPLETE: {passed}/{total} passed ({rate}%)",
    })
    r.lpush("metrics:recent_tasks", entry)
    r.ltrim("metrics:recent_tasks", 0, 49)
    print(f"\n{'=' * 60}")
    print(f"  RUN COMPLETE: {passed}/{total} passed ({rate}%)")
    print(f"  Pushed to dashboard.")
    print(f"{'=' * 60}")
    return True


def push_to_redis(r, task):
    today = datetime.now(timezone.utc).date().isoformat()
    daily_key = f"metrics:daily:{today}"

    r.hincrby(daily_key, "tasks_total", 1)
    if task["result"] == "PASS":
        r.hincrby(daily_key, "tasks_success", 1)

    est_duration_ms = int(task["tokens"] / 45.0 * 1000)

    display_phase = "phase3_exhausted" if task["phase"] == "none" else task["phase"]
    attempts = 3 if task["phase"] == "none" else 1

    entry = json.dumps({
        "task_id": task["task_id"],
        "type": f"v3_phase3_stress ({display_phase})",
        "success": task["result"] == "PASS",
        "attempts": attempts,
        "duration_ms": est_duration_ms,
        "tokens": task["tokens"],
        "model": "qwen3:14b",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "progress": f"{task['idx']}/{task['total']}",
        "rate_tasks_hr": task["rate"],
    })
    r.lpush("metrics:recent_tasks", entry)
    r.ltrim("metrics:recent_tasks", 0, 49)

    status = "PASS" if task["result"] == "PASS" else "FAIL"
    print(f"  -> Redis: [{task['idx']}/{task['total']}] "
          f"{task['task_id']}: {status} (via {task['phase']})")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v3_phase3_stress.log"
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {redis_url}")
    print(f"Watching: {log_path}")
    print(f"Dashboard: http://localhost:30001")
    print()

    seen = set()

    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                task = parse_line(line)
                if task and task["task_id"] not in seen:
                    seen.add(task["task_id"])
                    push_to_redis(r, task)
        if seen:
            print(f"\nBackfilled {len(seen)} tasks. Watching for new...\n")

    with open(log_path) as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                task = parse_line(line)
                if task and task["task_id"] not in seen:
                    seen.add(task["task_id"])
                    push_to_redis(r, task)
                if check_completion(line, r):
                    print("Bridge exiting.")
                    return
            else:
                time.sleep(1)


if __name__ == "__main__":
    main()

"""Bridge V3 benchmark log output to ATLAS Dashboard via Redis.

Usage:
    python3 v3_dashboard_bridge.py /tmp/v3_validation.log
    python3 v3_dashboard_bridge.py /tmp/v3_validation.log --backfill

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


def push_to_redis(r, task):
    today = datetime.now(timezone.utc).date().isoformat()
    daily_key = f"metrics:daily:{today}"

    # Increment daily counters
    r.hincrby(daily_key, "tasks_total", 1)
    if task["result"] == "PASS":
        r.hincrby(daily_key, "tasks_success", 1)

    # Push to recent tasks list (dashboard shows last 20)
    entry = json.dumps({
        "task_id": task["task_id"],
        "status": "success" if task["result"] == "PASS" else "failed",
        "phase": task["phase"],
        "tokens": task["tokens"],
        "rate_tasks_hr": task["rate"],
        "progress": f"{task['idx']}/{task['total']}",
        "source": "v3_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    r.lpush("metrics:recent_tasks", entry)
    r.ltrim("metrics:recent_tasks", 0, 49)  # keep last 50

    status = "PASS" if task["result"] == "PASS" else "FAIL"
    print(f"  -> Redis: [{task['idx']}/{task['total']}] "
          f"{task['task_id']}: {status}")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v3_validation.log"
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {redis_url}")
    print(f"Watching: {log_path}")
    print(f"Dashboard: http://localhost:30001")
    print()

    seen = set()

    # Backfill existing lines
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                task = parse_line(line)
                if task and task["task_id"] not in seen:
                    seen.add(task["task_id"])
                    push_to_redis(r, task)
        if seen:
            print(f"\nBackfilled {len(seen)} tasks. Watching for new...\n")

    # Tail for new results
    with open(log_path) as f:
        f.seek(0, 2)  # seek to end
        while True:
            line = f.readline()
            if line:
                task = parse_line(line)
                if task and task["task_id"] not in seen:
                    seen.add(task["task_id"])
                    push_to_redis(r, task)
            else:
                time.sleep(1)


if __name__ == "__main__":
    main()

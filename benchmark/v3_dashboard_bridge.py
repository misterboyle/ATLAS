#!/usr/bin/env python3
"""ATLAS V3 Dashboard Bridge.

Tails a V3 benchmark runner log file, parses task results, and pushes
metrics to Redis for the ATLAS Dashboard to display.

Usage:
    PYTHONUNBUFFERED=1 .venv/bin/python benchmark/v3_dashboard_bridge.py /tmp/run.log

    # Post-hoc (non-tailing):
    .venv/bin/python benchmark/v3_dashboard_bridge.py --no-tail /tmp/run.log

    # Dry run (no Redis, print parsed results):
    .venv/bin/python benchmark/v3_dashboard_bridge.py --dry-run /tmp/run.log

Redis keys written:
    atlas:run:<run_id>:tasks      LIST of JSON task results
    atlas:run:<run_id>:summary    HASH {total,passed,failed,pass_rate,last_update}
    atlas:run:<run_id>:phases     HASH {phase1:N, pr_cot:N, ...}
    atlas:metrics:daily:<date>    HASH {total,passed,failed}
    atlas:metrics:recent_tasks    LIST of last 100 task results (all runs)
    atlas:metrics:active_run      STRING current run_id

Requires: pip install redis
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone


def parse_task_line(line):
    """Parse V3 runner task line.

    Format: [N/M] task_id: PASS|FAIL (via phase, T tok) [R tasks/hr]
    """
    m = re.match(
        r'\s*\[(\d+)/(\d+)\]\s+(.+?):\s+(PASS|FAIL)\s+'
        r'\(via\s+(\w+),\s+(\d+)\s+tok\)\s+'
        r'\[(\d+)\s+tasks/hr\]',
        line,
    )
    if not m:
        return None
    return {
        "index": int(m.group(1)),
        "total": int(m.group(2)),
        "task_id": m.group(3),
        "status": m.group(4),
        "phase": m.group(5),
        "tokens": int(m.group(6)),
        "rate": int(m.group(7)),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def parse_run_id(line):
    """Extract run ID from header."""
    m = re.search(r'Run ID:\s+(\S+)', line)
    return m.group(1) if m else None


def parse_summary(line):
    """Parse final summary line."""
    m = re.match(r'\s*pass@1:\s+(\d+)/(\d+)\s+\((\d+\.\d+)%\)', line)
    if not m:
        return None
    return {
        "passed": int(m.group(1)),
        "total": int(m.group(2)),
        "pass_rate": float(m.group(3)) / 100,
    }


class RedisBridge:
    """Push metrics to Redis."""

    def __init__(self, host="localhost", port=6379):
        import redis
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.r.ping()

    def push_task(self, run_id, td):
        date_key = datetime.now().strftime("%Y-%m-%d")
        pipe = self.r.pipeline()
        # Per-run task list
        pipe.rpush(f"atlas:run:{run_id}:tasks", json.dumps(td))
        # Per-run summary
        pipe.hincrby(f"atlas:run:{run_id}:summary", "total", 1)
        status_key = "passed" if td["status"] == "PASS" else "failed"
        pipe.hincrby(f"atlas:run:{run_id}:summary", status_key, 1)
        pipe.hset(f"atlas:run:{run_id}:summary", "last_update", td["ts"])
        # Phase breakdown (passes only)
        if td["status"] == "PASS":
            pipe.hincrby(f"atlas:run:{run_id}:phases", td["phase"], 1)
        # Daily metrics
        pipe.hincrby(f"atlas:metrics:daily:{date_key}", "total", 1)
        pipe.hincrby(f"atlas:metrics:daily:{date_key}", status_key, 1)
        # Recent tasks (last 100)
        pipe.lpush("atlas:metrics:recent_tasks", json.dumps(td))
        pipe.ltrim("atlas:metrics:recent_tasks", 0, 99)
        # Active run
        pipe.set("atlas:metrics:active_run", run_id)
        pipe.execute()
        # Recompute pass rate
        s = self.r.hgetall(f"atlas:run:{run_id}:summary")
        tot = int(s.get("total", 0))
        pas = int(s.get("passed", 0))
        if tot > 0:
            self.r.hset(f"atlas:run:{run_id}:summary", "pass_rate",
                        f"{pas/tot:.4f}")

    def set_complete(self, run_id):
        self.r.hset(f"atlas:run:{run_id}:summary", "status", "complete")
        self.r.delete("atlas:metrics:active_run")


class StdoutBridge:
    """Fallback: print parsed results (no Redis)."""

    def push_task(self, run_id, td):
        print(f"[BRIDGE] {run_id}: {td['task_id']} "
              f"{td['status']} (via {td['phase']})")

    def set_complete(self, run_id):
        print(f"[BRIDGE] Run {run_id} complete")


def tail_file(path, poll=1.0):
    """Yield lines, tailing like tail -f."""
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(poll)


def read_file(path):
    """Yield all lines (non-tailing)."""
    with open(path, "r") as f:
        for line in f:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser(
        description="ATLAS V3 Dashboard Bridge"
    )
    ap.add_argument("logfile", help="Path to V3 runner log file")
    ap.add_argument("--no-tail", action="store_true",
                    help="Process file once without tailing")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print parsed results without Redis")
    args = ap.parse_args()

    if args.dry_run:
        bridge = StdoutBridge()
    else:
        try:
            bridge = RedisBridge(args.redis_host, args.redis_port)
            print(f"[BRIDGE] Connected to Redis "
                  f"{args.redis_host}:{args.redis_port}")
        except Exception as e:
            print(f"[BRIDGE] Redis failed: {e}", file=sys.stderr)
            print("[BRIDGE] Falling back to stdout", file=sys.stderr)
            bridge = StdoutBridge()

    run_id = None
    count = 0
    lines = read_file(args.logfile) if args.no_tail else tail_file(args.logfile)
    mode = "one-shot" if args.no_tail else "tailing"
    print(f"[BRIDGE] Processing {args.logfile} ({mode})")

    try:
        for line in lines:
            if run_id is None:
                rid = parse_run_id(line)
                if rid:
                    run_id = rid
                    print(f"[BRIDGE] Run ID: {run_id}")
                    continue

            td = parse_task_line(line)
            if td and run_id:
                bridge.push_task(run_id, td)
                count += 1
                continue

            s = parse_summary(line)
            if s and run_id:
                bridge.set_complete(run_id)
                print(f"[BRIDGE] Complete: {s['passed']}/{s['total']} "
                      f"({s['pass_rate']:.1%})")
                if args.no_tail:
                    break
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"[BRIDGE] Not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    print(f"[BRIDGE] Processed {count} tasks")


if __name__ == "__main__":
    main()

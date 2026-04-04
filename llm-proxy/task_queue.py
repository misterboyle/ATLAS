"""Redis sorted-set priority queue for llm-proxy.

Priority levels:
    P0 (critical)  -- score band 0*1e13 .. 1*1e13-1
    P1 (normal)    -- score band 1*1e13 .. 2*1e13-1
    P2 (low)       -- score band 2*1e13 .. 3*1e13-1

Within each band, tasks are ordered by enqueue time (FIFO).
Task payloads are stored in a companion Redis hash.
"""

from __future__ import annotations

import json
import os
import time
from enum import IntEnum
from typing import Any, Optional

import redis


QUEUE_KEY = "llm_proxy:task_queue"
PAYLOAD_PREFIX = "llm_proxy:task:"
STATUS_PREFIX = "llm_proxy:task_status:"
BAND_WIDTH = 10**13  # must exceed max epoch-ms (~1.78e12 in 2026)


class Priority(IntEnum):
    """Task priority levels (lower numeric value == higher priority)."""
    P0 = 0  # critical / real-time
    P1 = 1  # normal
    P2 = 2  # background / best-effort


def _score(priority: Priority, timestamp_ms: Optional[int] = None) -> float:
    """Compute sorted-set score from priority + wallclock."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return float(priority * BAND_WIDTH + timestamp_ms)


def _priority_range(priority: Priority) -> tuple[float, float]:
    lo = priority * BAND_WIDTH
    hi = (priority + 1) * BAND_WIDTH - 1
    return (lo, hi)


class TaskQueue:
    """Redis-backed priority queue using a sorted set."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        queue_key: str = QUEUE_KEY,
    ) -> None:
        if redis_client is not None:
            self._r = redis_client
        else:
            self._r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        self._queue_key = queue_key

    def enqueue_task(
        self,
        task_id: str,
        priority: Priority | int,
        payload: dict[str, Any] | None = None,
    ) -> float:
        """Add a task to the queue. Returns the score assigned."""
        priority = Priority(int(priority))
        score = _score(priority)
        pipe = self._r.pipeline()
        pipe.zadd(self._queue_key, {task_id: score})
        if payload is not None:
            pipe.set(f"{PAYLOAD_PREFIX}{task_id}", json.dumps(payload))
        pipe.execute()
        self.set_task_status(task_id, "pending")
        return score

    def dequeue_task(self) -> Optional[dict[str, Any]]:
        """Pop the highest-priority (lowest-score) task, or None."""
        result = self._r.zpopmin(self._queue_key, count=1)
        if not result:
            return None
        task_id, score = result[0]
        score = float(score)
        priority_int = int(score // BAND_WIDTH)
        priority = Priority(priority_int)
        payload_key = f"{PAYLOAD_PREFIX}{task_id}"
        raw = self._r.get(payload_key)
        payload = json.loads(raw) if raw else None
        if raw is not None:
            self._r.delete(payload_key)
        return {
            "task_id": task_id,
            "priority": priority.name,
            "priority_value": priority.value,
            "score": score,
            "payload": payload,
        }

    def get_queue_depths(self) -> dict[str, int]:
        """Return queued task counts per priority level."""
        depths: dict[str, int] = {}
        total = 0
        for p in Priority:
            lo, hi = _priority_range(p)
            count = self._r.zcount(self._queue_key, lo, hi)
            depths[p.name] = count
            total += count
        depths["total"] = total
        return depths

    def peek(self, count: int = 5) -> list[dict[str, Any]]:
        """Preview the next *count* tasks without removing them."""
        items = self._r.zrange(self._queue_key, 0, count - 1, withscores=True)
        results = []
        for task_id, score in items:
            score = float(score)
            priority_int = int(score // BAND_WIDTH)
            results.append({
                "task_id": task_id,
                "priority": Priority(priority_int).name,
                "score": score,
            })
        return results

    def remove_task(self, task_id: str) -> bool:
        """Remove a specific task from the queue."""
        pipe = self._r.pipeline()
        pipe.zrem(self._queue_key, task_id)
        pipe.delete(f"{PAYLOAD_PREFIX}{task_id}")
        removed, _ = pipe.execute()
        return removed > 0

    def flush(self) -> int:
        """Remove all tasks. Returns the number removed."""
        count = self._r.zcard(self._queue_key)
        members = self._r.zrange(self._queue_key, 0, -1)
        pipe = self._r.pipeline()
        pipe.delete(self._queue_key)
        for m in members:
            pipe.delete(f"{PAYLOAD_PREFIX}{m}")
        pipe.execute()
        return count

    def set_task_status(
        self,
        task_id: str,
        status: str,
        result: Any = None,
    ) -> None:
        """Set task status. Statuses: pending, processing, complete, failed."""
        key = f"{STATUS_PREFIX}{task_id}"
        mapping: dict[str, str] = {
            "status": status,
            "updated_at": str(time.time()),
        }
        if result is not None:
            mapping["result"] = json.dumps(result)
        self._r.hset(key, mapping=mapping)
        # Auto-expire completed/failed status after 24 hours
        if status in ("complete", "failed"):
            self._r.expire(key, 86400)

    def get_task_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get task status and optional result. Returns None if unknown."""
        key = f"{STATUS_PREFIX}{task_id}"
        data = self._r.hgetall(key)
        if not data:
            return None
        out: dict[str, Any] = {"status": data.get("status", "unknown")}
        if "result" in data:
            try:
                out["result"] = json.loads(data["result"])
            except (json.JSONDecodeError, TypeError):
                out["result"] = data["result"]
        if "updated_at" in data:
            out["updated_at"] = float(data["updated_at"])
        return out

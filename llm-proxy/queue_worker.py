"""Queue worker -- consumes V3 tasks from Redis priority queue.

Polls dequeue_task() every WORKER_POLL_INTERVAL seconds. For each task:
  1. Sets status to processing
  2. POSTs to rag-api /v3/run
  3. Stores result via set_task_status()
  4. Logs metrics to Redis daily counters

Run standalone: python queue_worker.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone

import httpx
import redis

from task_queue import TaskQueue

RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-api:8001")
POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "1.0"))
V3_TIMEOUT = float(os.getenv("V3_TIMEOUT", "600"))

_redis = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
)
queue = TaskQueue(redis_client=_redis)


def log_metrics(
    request_type: str, model: str, tokens: int,
    success: bool, duration_ms: int, attempts: int = 1, phase: str = "",
) -> None:
    """Write metrics to Redis (mirrors main.py log_metrics)."""
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        dk = f"atlas:metrics:daily:{today}"
        _redis.hincrby(dk, "tasks_total", 1)
        _redis.hincrby(dk, "requests_total", 1)
        _redis.hincrby(dk, "tasks_success" if success else "tasks_failed", 1)
        _redis.hincrby(dk, "total_tokens", tokens)
        _redis.hincrby(dk, "total_duration_ms", duration_ms)
        _redis.hincrby(dk, "total_attempts", attempts)
        record = json.dumps({
            "type": request_type, "model": model, "success": success,
            "attempts": attempts, "duration_ms": duration_ms,
            "tokens": tokens, "phase": phase,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _redis.lpush("atlas:metrics:recent_tasks", record)
        _redis.ltrim("atlas:metrics:recent_tasks", 0, 99)
    except Exception as exc:
        print(f"[queue_worker] metrics error: {exc}")


async def process_task(task_data: dict) -> None:
    """Send a dequeued task to rag-api /v3/run and record the outcome."""
    task_id: str = task_data["task_id"]
    payload: dict = task_data.get("payload") or {}
    queue.set_task_status(task_id, "processing")
    prompt = payload.get("prompt", "")
    mode = payload.get("mode", "thorough")
    timeout_s = payload.get("timeout_seconds", 600)
    model = payload.get("model", "atlas-v3")
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=V3_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_API_URL}/v3/run",
                json={"task_id": task_id, "prompt": prompt,
                      "mode": mode, "timeout_seconds": timeout_s},
            )
        duration_ms = int((time.time() - start) * 1000)
        if resp.status_code == 200:
            result = resp.json()
            queue.set_task_status(task_id, "complete", result=result)
            log_metrics("v3_queue", model, result.get("total_tokens", 0),
                        True, duration_ms,
                        attempts=result.get("candidates_generated", 1),
                        phase=result.get("phase_solved", ""))
            print(f"[queue_worker] {task_id} complete ({duration_ms}ms)")
        else:
            err = {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}
            queue.set_task_status(task_id, "failed", result=err)
            log_metrics("v3_queue", model, 0, False, duration_ms)
            print(f"[queue_worker] {task_id} failed: HTTP {resp.status_code}")
    except Exception as exc:
        duration_ms = int((time.time() - start) * 1000)
        queue.set_task_status(
            task_id, "failed",
            result={"error": type(exc).__name__, "detail": str(exc)[:500]})
        log_metrics("v3_queue", model, 0, False, duration_ms)
        print(f"[queue_worker] {task_id} error: {exc}")


async def worker_loop() -> None:
    """Poll the priority queue and process tasks sequentially."""
    print(f"[queue_worker] started rag_api={RAG_API_URL} poll={POLL_INTERVAL}s")
    while True:
        try:
            task_data = queue.dequeue_task()
            if task_data is not None:
                print(f"[queue_worker] dequeued {task_data['task_id']} "
                      f"(priority={task_data['priority']})")
                await process_task(task_data)
            else:
                await asyncio.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("[queue_worker] shutting down")
            break
        except Exception as exc:
            print(f"[queue_worker] loop error: {exc}")
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(worker_loop())

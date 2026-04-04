"""
ATLAS Dashboard - Monitoring and control interface.
"""

import os
import redis
import json
from datetime import date, datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="ATLAS Dashboard")

templates = Jinja2Templates(directory="templates")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# llm-proxy sorted-set priority queue constants
QUEUE_KEY = "llm_proxy:task_queue"
BAND_WIDTH = 10**13

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    # Get queue stats from sorted-set priority queue
    queue_stats = {
        "p0": redis_client.zcount(QUEUE_KEY, 0, BAND_WIDTH - 1),
        "p1": redis_client.zcount(QUEUE_KEY, BAND_WIDTH, 2 * BAND_WIDTH - 1),
        "p2": redis_client.zcount(QUEUE_KEY, 2 * BAND_WIDTH, 3 * BAND_WIDTH - 1),
    }

    # Get today's metrics (use UTC to match task-worker)
    today = datetime.now(timezone.utc).date().isoformat()
    daily_stats = redis_client.hgetall(f"atlas:metrics:daily:{today}")

    # Get recent tasks
    recent_tasks = redis_client.lrange("atlas:metrics:recent_tasks", 0, 19)
    recent_tasks = [json.loads(t) for t in recent_tasks]

    # Get recent validation results (from task-worker POST)
    validation_results = redis_client.lrange("atlas:validation:results", 0, 9)
    validation_results = [json.loads(r) for r in validation_results]

    # Get weekly trend (use UTC)
    weekly_trend = []
    utc_today = datetime.now(timezone.utc).date()
    for i in range(7):
        day = (utc_today - timedelta(days=i)).isoformat()
        stats = redis_client.hgetall(f"atlas:metrics:daily:{day}")
        weekly_trend.append({
            "date": day,
            "total": int(stats.get("tasks_total", 0)),
            "success": int(stats.get("tasks_success", 0))
        })
    weekly_trend.reverse()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "queue_stats": queue_stats,
            "daily_stats": daily_stats,
            "recent_tasks": recent_tasks,
            "validation_results": validation_results,
            "weekly_trend": weekly_trend,
        }
    )

@app.get("/api/stats")
async def api_stats():
    """API endpoint for stats (for AJAX refresh)."""
    today = datetime.now(timezone.utc).date().isoformat()

    return {
        "queue": {
            "p0": redis_client.zcount(QUEUE_KEY, 0, BAND_WIDTH - 1),
            "p1": redis_client.zcount(QUEUE_KEY, BAND_WIDTH, 2 * BAND_WIDTH - 1),
            "p2": redis_client.zcount(QUEUE_KEY, 2 * BAND_WIDTH, 3 * BAND_WIDTH - 1),
        },
        "daily": redis_client.hgetall(f"atlas:metrics:daily:{today}"),
        "recent": [
            json.loads(t)
            for t in redis_client.lrange("atlas:metrics:recent_tasks", 0, 9)
        ],
        "validation": [
            json.loads(r)
            for r in redis_client.lrange("atlas:validation:results", 0, 9)
        ]
    }

@app.post("/api/validation/results")
async def receive_validation_results(request: Request):
    """Receive task results from task-worker for dashboard visibility (pardot-jetb)."""
    try:
        payload = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    task_id = payload.get("task_id", "unknown")

    # Store full result for dashboard display
    result_record = json.dumps({
        "task_id": task_id,
        "type": payload.get("type", "coding"),
        "status": payload.get("status", "unknown"),
        "success": payload.get("success", False),
        "result": payload.get("result"),
        "metrics": payload.get("metrics"),
        "completed_at": payload.get("completed_at"),
        "received_at": datetime.now(timezone.utc).isoformat(),
    })
    redis_client.lpush("atlas:validation:results", result_record)
    redis_client.ltrim("atlas:validation:results", 0, 999)

    return {"status": "ok", "task_id": task_id}

@app.get("/api/validation/results")
async def get_validation_results():
    """Get recent validation results for dashboard display."""
    results = redis_client.lrange("atlas:validation:results", 0, 19)
    return [json.loads(r) for r in results]

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)

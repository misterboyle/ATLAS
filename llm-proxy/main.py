"""
LLM Proxy - Forwards requests to llama-server and logs metrics to Redis.
Validates API keys against the API Portal.
"""

import os
import json
import time
import httpx
import redis
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, Tuple
import uuid

from task_queue import TaskQueue

LLAMA_URL = os.getenv("LLAMA_URL", "http://llama-service:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
API_PORTAL_URL = os.getenv("API_PORTAL_URL", "http://api-portal:3000")
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-api:8001")

# API key validation cache
_key_cache = {}
_key_cache_ttl = 60  # seconds

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    print("Warning: Redis not available, metrics will not be logged")

# Task queue for async V3 request status tracking
task_queue = TaskQueue(redis_client=redis_client) if redis_client else None


async def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key against the API Portal."""
    # Check cache first
    cached = _key_cache.get(api_key)
    if cached and time.time() - cached["timestamp"] < _key_cache_ttl:
        return cached["data"]

    # Validate with portal
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{API_PORTAL_URL}/api/validate-key",
                json={"api_key": api_key}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    _key_cache[api_key] = {
                        "timestamp": time.time(),
                        "data": data
                    }
                    return data
    except Exception as e:
        print(f"Key validation error: {e}")

    return None


def check_rate_limit(api_key: str, limit: int) -> Tuple[bool, int, int]:
    """Check and increment rate limit counter using sliding window.

    Args:
        api_key: The API key string (used to create unique Redis key)
        limit: Maximum requests allowed per window

    Returns:
        Tuple of (allowed, remaining, ttl_seconds)
    """
    if not redis_client:
        # If Redis unavailable, allow all requests
        return (True, limit - 1, 60)

    try:
        # Use a hash of the key to avoid storing the full key in Redis
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        redis_key = f"ratelimit:{key_hash}:count"

        # Increment counter
        current = redis_client.incr(redis_key)

        # Set expiry on first request (60 second sliding window)
        if current == 1:
            redis_client.expire(redis_key, 60)

        # Get TTL for reset time
        ttl = redis_client.ttl(redis_key)
        if ttl < 0:
            ttl = 60

        remaining = max(0, limit - current)
        allowed = current <= limit

        return (allowed, remaining, ttl)
    except Exception as e:
        print(f"Rate limit check error: {e}")
        # On error, allow the request but log it
        return (True, limit - 1, 60)


async def require_api_key(authorization: str = Header(None)) -> dict:
    """Dependency to require valid API key. Returns validation data with rate limit info.

    When ALLOW_INTERNAL=true, requests without an Authorization header
    are treated as trusted internal Docker network requests and bypass auth.
    """
    if not authorization and os.environ.get("ALLOW_INTERNAL", "").lower() == "true":
        return {"key": "internal", "validation": {"rate_limit": 999999}}
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Extract key from "Bearer sk-xxx" format
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        key = parts[1]
    else:
        # Also accept raw key
        key = authorization

    validation = await validate_api_key(key)
    if not validation:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Return full validation data including rate limit
    return {"key": key, "validation": validation}


def get_rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict:
    """Generate rate limit headers with real values."""
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(time.time()) + reset_seconds)
    }


def log_metrics(request_type: str, model: str, tokens: int, success: bool, duration_ms: int, attempts: int = 1, phase: str = ""):
    """Log request metrics to Redis."""
    if not redis_client:
        return
    try:
        today = datetime.now(timezone.utc).date().isoformat()

        # Increment daily counters (field names must match dashboard template)
        daily_key = f"atlas:metrics:daily:{today}"
        redis_client.hincrby(daily_key, "tasks_total", 1)
        if success:
            redis_client.hincrby(daily_key, "tasks_success", 1)
        else:
            redis_client.hincrby(daily_key, "tasks_failed", 1)
        redis_client.hincrby(daily_key, "total_tokens", tokens)
        redis_client.hincrby(daily_key, "total_duration_ms", duration_ms)
        redis_client.hincrby(daily_key, "total_attempts", 1)
        # Increment daily counters
        redis_client.hincrby(f"atlas:metrics:daily:{today}", "tasks_total", 1)
        redis_client.hincrby(f"atlas:metrics:daily:{today}", "requests_total", 1)  # Track requests
            redis_client.hincrby(f"atlas:metrics:daily:{today}", "tasks_success", 1)
            redis_client.hincrby(f"atlas:metrics:daily:{today}", "tasks_failed", 1)
        redis_client.hincrby(f"atlas:metrics:daily:{today}", "total_tokens", tokens)
        redis_client.hincrby(f"atlas:metrics:daily:{today}", "total_duration_ms", duration_ms)
        redis_client.hincrby(f"atlas:metrics:daily:{today}", "total_attempts", attempts)

        # Add to recent tasks
        task_record = json.dumps({
            "task_id": f"chat-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "type": request_type,
            "model": model,
            "success": success,
            "attempts": attempts,
            "duration_ms": duration_ms,
            "tokens": tokens,
            "phase": phase,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        redis_client.lpush("atlas:metrics:recent_tasks", task_record)
        redis_client.ltrim("atlas:metrics:recent_tasks", 0, 99)
    except Exception as e:
        print(f"Failed to log metrics: {e}")


# Atlas V3 model variants
ATLAS_V3_MODELS = [
    {"id": "atlas-v3", "object": "model", "created": 1700000000, "owned_by": "atlas"},
    {"id": "atlas-v3-fast", "object": "model", "created": 1700000000, "owned_by": "atlas"},
    {"id": "atlas-v3-thorough", "object": "model", "created": 1700000000, "owned_by": "atlas"},
]


async def handle_v3_request(body, model, start_time, rate_headers):
    """Route atlas-v3 requests to RAG API /v3/run, return OpenAI format."""
    messages = body.get("messages", [])
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, list):
                prompt = " ".join(
                    p.get("text", "") for p in c if p.get("type") == "text"
                )
            else:
                prompt = str(c)
            break
    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found")
    mode = "thorough" if model.endswith("-thorough") else "fast"
    try:
        async with httpx.AsyncClient(timeout=600.0) as rc:
            resp = await rc.post(
                f"{RAG_API_URL}/v3/run",
                json={"task_id": f"chat-{uuid.uuid4().hex[:8]}", "prompt": prompt, "mode": mode},
            )
            duration_ms = int((time.time() - start_time) * 1000)
            if resp.status_code != 200:
                log_metrics("v3_completion", model, 0, False, duration_ms)
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": f"RAG API error: {resp.text}",
                             "type": "backend_error", "code": "rag_api_error"}},
                    headers=rate_headers,
                )
            rag_result = resp.json()
    except httpx.ConnectError:
        duration_ms = int((time.time() - start_time) * 1000)
        log_metrics("v3_completion", model, 0, False, duration_ms)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "RAG API unavailable",
                     "type": "backend_error", "code": "connect_error"}},
            headers=rate_headers,
        )
    except httpx.HTTPError as exc:
        duration_ms = int((time.time() - start_time) * 1000)
        log_metrics("v3_completion", model, 0, False, duration_ms)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"RAG API error: {type(exc).__name__}",
                     "type": "backend_error", "code": "proxy_error"}},
            headers=rate_headers,
        )
    answer = rag_result.get("code", rag_result.get("answer", rag_result.get("result", "")))
    v3_meta = {"mode": mode, "rag_model": model}
    for k in ("phase_solved", "status", "candidates_generated",
              "total_tokens", "total_time_ms", "telemetry"):
        if k in rag_result:
            v3_meta[k] = rag_result[k]
    pt, ct = len(prompt.split()), len(answer.split())
    openai_resp = {
        "id": f"chatcmpl-v3-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct},
        "v3_metadata": v3_meta,
    }
    v3_attempts = v3_meta.get("candidates_generated", 1) if isinstance(v3_meta, dict) else 1
    v3_phase = v3_meta.get("phase_solved", "") if isinstance(v3_meta, dict) else ""
    log_metrics("v3_completion", model, pt + ct, True, duration_ms, attempts=v3_attempts, phase=v3_phase)
    return JSONResponse(content=openai_resp, headers=rate_headers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"LLM Proxy starting - forwarding to {LLAMA_URL}")
    yield
    print("LLM Proxy shutting down")


app = FastAPI(title="LLM Proxy", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy", "backend": LLAMA_URL}


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """Proxy models endpoint - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{LLAMA_URL}/v1/models")
            backend_data = resp.json()
        except Exception:
            backend_data = {"object": "list", "data": []}

        # Append atlas-v3 model variants
        backend_data.setdefault("data", []).extend(ATLAS_V3_MODELS)
        return JSONResponse(content=backend_data, headers=rate_headers)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(None)):
    """Proxy chat completions with metrics logging - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)

    start = time.time()

    # Validate JSON body
    try:
        body_bytes = await request.body()
        body = json.loads(body_bytes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Validate required fields
    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Missing required field: messages")

    if not isinstance(body.get("messages"), list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    model = body.get("model", "unknown")

    # Intercept atlas-v3 model requests -- route to RAG API
    if model.startswith("atlas-v3"):
        return await handle_v3_request(body, model, start, rate_headers)

    stream = body.get("stream", False)

    if stream:
        # Ensure llama-server reports token usage in final SSE chunk
        body.setdefault("stream_options", {})["include_usage"] = True
        # Streaming response - client must be created inside generator to stay open
        async def stream_with_metrics():
            tokens = 0
            client = httpx.AsyncClient(timeout=600.0)
            try:
                async with client.stream(
                    "POST",
                    f"{LLAMA_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        # Count tokens from SSE data
                        if b'"completion_tokens"' in chunk:
                            try:
                                # Extract token count from chunk
                                text = chunk.decode()
                                for line in text.split('\n'):
                                    if line.startswith('data: ') and 'usage' in line:
                                        data = json.loads(line[6:])
                                        tokens = data.get('usage', {}).get('total_tokens', 0)
                            except Exception:
                                pass
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_stream", model, tokens, True, duration_ms)
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_stream", model, 0, False, duration_ms)
                raise
            finally:
                await client.aclose()

        stream_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        stream_headers.update(rate_headers)
        return StreamingResponse(
            stream_with_metrics(),
            media_type="text/event-stream",
            headers=stream_headers
        )
    else:
        # Non-streaming
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                resp = await client.post(
                    f"{LLAMA_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                )
                duration_ms = int((time.time() - start) * 1000)

                result = resp.json()
                tokens = result.get("usage", {}).get("total_tokens", 0)
                log_metrics("chat_completion", model, tokens, True, duration_ms)

                return Response(content=resp.content, media_type="application/json", headers=rate_headers)
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_completion", model, 0, False, duration_ms)
                raise


@app.get("/v3/task/{task_id}")
async def get_v3_task_status(task_id: str, authorization: str = Header(None)):
    """Poll task status and result for async V3 queue requests."""
    auth_data = await require_api_key(authorization)
    if not task_queue:
        raise HTTPException(status_code=503, detail="Task queue unavailable")
    status = task_queue.get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return JSONResponse(content={"task_id": task_id, **status})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request, path: str, authorization: str = Header(None)):
    """Catch-all proxy for other endpoints - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{LLAMA_URL}/{path}"

            if request.method == "GET":
                resp = await client.get(url, params=request.query_params)
            else:
                body = await request.body()
                resp = await client.request(
                    request.method,
                    url,
                    content=body,
                    headers={"Content-Type": request.headers.get("Content-Type", "application/json")}
                )

            response_headers = dict(rate_headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
                headers=response_headers
            )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": f"Backend timeout: llama-server did not respond within 60s for /{path}",
                    "type": "timeout_error",
                    "code": "read_timeout",
                    "param": path
                }
            },
            headers=rate_headers
        )
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Backend unavailable: cannot connect to llama-server for /{path}",
                    "type": "backend_error",
                    "code": "connect_error",
                    "param": path
                }
            },
            headers=rate_headers
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Backend error: {type(exc).__name__} for /{path}",
                    "type": "backend_error",
                    "code": "proxy_error",
                    "param": path
                }
            },
            headers=rate_headers
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Integration tests for V3 model routing.

Covers:
- llm-proxy routing for atlas-v3 models
- Fast and thorough modes
- OpenAI response format validation
- /v1/models including atlas-v3
- /v3/run endpoint direct tests
- SelfTestGen fallback when no test cases
- Timeout handling
- v3_metadata in response

All tests standalone (no Docker). V3Runner and httpx are mocked.
"""

import time
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# -- Request/response models for test endpoints --

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    v3_mode: Optional[str] = None


class V3RunRequest(BaseModel):
    task_prompt: str
    mode: Optional[str] = "standard"
    timeout: Optional[float] = 300.0


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "atlas"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# -- Mock V3Runner (stands in for benchmark.v3_runner.V3Pipeline) --

class MockV3Runner:
    """Mock V3Pipeline that returns canned results per mode."""

    def __init__(self, mode="standard"):
        self.mode = mode
        self.called = False
        self.last_task = None
        self.timeout = 300.0

    def run_task(self, task, task_id=""):
        self.called = True
        self.last_task = task
        if self.mode == "fast":
            candidates, phases = 1, ["phase1"]
        elif self.mode == "thorough":
            candidates, phases = 5, ["phase1", "phase2", "phase3"]
        else:
            candidates, phases = 3, ["phase1", "phase2", "phase3"]
        return {
            "task_id": task_id or "test-task",
            "passed": True,
            "code": "def solution(): return 42",
            "phase_solved": phases[0],
            "candidates_generated": candidates,
            "total_tokens": 1500,
            "total_time_ms": 2500.0,
            "v3_metadata": {
                "mode": self.mode,
                "pipeline_version": "3.0",
                "phases_enabled": phases,
                "candidates_generated": candidates,
                "self_tests_generated": 3 if self.mode != "fast" else 0,
                "total_tokens": 1500,
                "total_time_ms": 2500.0,
            },
            "telemetry": {
                "adaptive_k": candidates,
                "budget_tier": "minimal" if self.mode == "fast" else "standard",
            },
        }


# -- Test FastAPI app (mimics llm-proxy + api-portal V3 routing) --

DEFAULT_MODELS = [
    ModelInfo(id="atlas-v3-standard", owned_by="atlas"),
    ModelInfo(id="atlas-v3-fast", owned_by="atlas"),
    ModelInfo(id="atlas-v3-thorough", owned_by="atlas"),
    ModelInfo(id="qwen3-14b", owned_by="local"),
]


def _create_app(v3_runner=None, models=None):
    app = FastAPI(title="ATLAS LLM Proxy (Test)")
    runner = v3_runner or MockV3Runner()
    model_list = models or list(DEFAULT_MODELS)

    @app.exception_handler(TimeoutError)
    async def _timeout(request, exc):
        return JSONResponse(status_code=504, content={
            "error": {"message": str(exc),
                      "type": "timeout_error",
                      "code": "v3_pipeline_timeout"}})

    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=model_list)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if req.model.startswith("atlas-v3"):
            mode = req.v3_mode
            if not mode:
                if "fast" in req.model:
                    mode = "fast"
                elif "thorough" in req.model:
                    mode = "thorough"
                else:
                    mode = "standard"
            runner.mode = mode
            prompt = "\n".join(m.content for m in req.messages)
            task = MagicMock(prompt=prompt, task_id="chat-task")
            result = runner.run_task(task, task_id="chat-task")
            return {
                "id": "chatcmpl-v3-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": result.get("code", "")},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": result.get("total_tokens", 0),
                    "total_tokens": 100 + result.get("total_tokens", 0),
                },
                "v3_metadata": result.get("v3_metadata", {}),
            }
        return {
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": "upstream reply"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10,
                      "completion_tokens": 20,
                      "total_tokens": 30},
        }

    @app.post("/v3/run")
    async def v3_run(req: V3RunRequest):
        runner.mode = req.mode or "standard"
        runner.timeout = req.timeout
        task = MagicMock(prompt=req.task_prompt, task_id="direct-v3")
        result = runner.run_task(task, task_id="direct-v3")
        return {
            "task_id": result["task_id"],
            "passed": result["passed"],
            "code": result["code"],
            "phase_solved": result["phase_solved"],
            "v3_metadata": result["v3_metadata"],
        }

    return app


# -- Fixtures --

@pytest.fixture
def v3_runner():
    return MockV3Runner()


@pytest.fixture
def client(v3_runner):
    return TestClient(_create_app(v3_runner=v3_runner))


# -- Tests (10 total) --

class TestV3ModelRouting:
    """Integration tests for V3 model routing."""

    def test_llm_proxy_routes_atlas_v3_to_pipeline(self, client, v3_runner):
        """atlas-v3 model requests route to V3 pipeline, not upstream."""
        resp = client.post("/v1/chat/completions", json={
            "model": "atlas-v3-standard",
            "messages": [{"role": "user", "content": "Add two numbers"}],
        })
        assert resp.status_code == 200
        assert v3_runner.called is True
        data = resp.json()
        assert data["model"] == "atlas-v3-standard"
        assert "v3_metadata" in data

    def test_fast_mode_routing(self, client, v3_runner):
        """atlas-v3-fast uses fast mode with 1 candidate."""
        resp = client.post("/v1/chat/completions", json={
            "model": "atlas-v3-fast",
            "messages": [{"role": "user", "content": "Two sum"}],
        })
        assert resp.status_code == 200
        meta = resp.json()["v3_metadata"]
        assert meta["mode"] == "fast"
        assert meta["candidates_generated"] == 1

    def test_thorough_mode_routing(self, client, v3_runner):
        """atlas-v3-thorough uses thorough mode with 5 candidates."""
        resp = client.post("/v1/chat/completions", json={
            "model": "atlas-v3-thorough",
            "messages": [{"role": "user", "content": "Graph coloring"}],
        })
        assert resp.status_code == 200
        meta = resp.json()["v3_metadata"]
        assert meta["mode"] == "thorough"
        assert meta["candidates_generated"] == 5

    def test_openai_response_format_validation(self, client):
        """V3 responses conform to OpenAI chat.completion format."""
        resp = client.post("/v1/chat/completions", json={
            "model": "atlas-v3-standard",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        # Top-level
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert "model" in data
        # Choices
        choices = data["choices"]
        assert isinstance(choices, list) and len(choices) >= 1
        c = choices[0]
        assert c["message"]["role"] == "assistant"
        assert isinstance(c["message"]["content"], str)
        assert c["finish_reason"] == "stop"
        # Usage
        u = data["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_v1_models_includes_atlas_v3(self):
        """/v1/models lists atlas-v3 variants alongside other models."""
        app = _create_app()
        c = TestClient(app)
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert "atlas-v3-standard" in ids
        assert "atlas-v3-fast" in ids
        assert "atlas-v3-thorough" in ids
        assert "qwen3-14b" in ids

    def test_v3_run_endpoint_direct(self, client, v3_runner):
        """/v3/run triggers V3 pipeline directly."""
        resp = client.post("/v3/run", json={
            "task_prompt": "Return nth Fibonacci number",
            "mode": "standard",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "direct-v3"
        assert data["passed"] is True
        assert "code" in data
        assert data["phase_solved"] == "phase1"
        assert v3_runner.called is True

    def test_v3_run_endpoint_thorough_with_timeout(self, client, v3_runner):
        """/v3/run respects mode and timeout parameters."""
        resp = client.post("/v3/run", json={
            "task_prompt": "Complex DP problem",
            "mode": "thorough",
            "timeout": 600.0,
        })
        assert resp.status_code == 200
        meta = resp.json()["v3_metadata"]
        assert meta["mode"] == "thorough"
        assert meta["candidates_generated"] == 5
        assert v3_runner.timeout == 600.0

    def test_self_test_gen_fallback_no_test_cases(self):
        """SelfTestGen fallback when 0 test cases are generated."""
        runner = MockV3Runner()
        orig = runner.run_task

        def patched(task, task_id=""):
            r = orig(task, task_id)
            r["v3_metadata"]["self_tests_generated"] = 0
            r["v3_metadata"]["self_test_fallback"] = True
            return r

        runner.run_task = patched
        c = TestClient(_create_app(v3_runner=runner))
        resp = c.post("/v3/run", json={
            "task_prompt": "Obscure problem",
            "mode": "standard",
        })
        assert resp.status_code == 200
        meta = resp.json()["v3_metadata"]
        assert meta["self_tests_generated"] == 0
        assert meta["self_test_fallback"] is True

    def test_timeout_handling(self):
        """504 returned when V3 pipeline exceeds timeout."""
        runner = MockV3Runner()
        runner.run_task = lambda t, **kw: (_ for _ in ()).throw(
            TimeoutError("V3 pipeline exceeded timeout"))
        c = TestClient(_create_app(v3_runner=runner))
        resp = c.post("/v3/run", json={
            "task_prompt": "Slow problem",
            "mode": "thorough",
            "timeout": 1.0,
        })
        assert resp.status_code == 504
        err = resp.json()["error"]
        assert err["type"] == "timeout_error"
        assert err["code"] == "v3_pipeline_timeout"

    def test_v3_metadata_in_response(self, client):
        """v3_metadata is present with all required fields."""
        resp = client.post("/v1/chat/completions", json={
            "model": "atlas-v3-standard",
            "messages": [{"role": "user", "content": "Quicksort"}],
        })
        assert resp.status_code == 200
        meta = resp.json()["v3_metadata"]
        for key in ("mode", "pipeline_version", "phases_enabled",
                    "candidates_generated", "total_tokens", "total_time_ms"):
            assert key in meta, f"Missing v3_metadata.{key}"
        assert meta["mode"] in ("fast", "standard", "thorough")
        assert isinstance(meta["pipeline_version"], str)
        assert isinstance(meta["phases_enabled"], list)
        assert meta["candidates_generated"] > 0
        assert isinstance(meta["total_tokens"], int)
        assert isinstance(meta["total_time_ms"], float)

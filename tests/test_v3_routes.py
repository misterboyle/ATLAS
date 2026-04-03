"""Tests for POST /v3/run endpoint (v3_routes.py)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_RAG_API = str(Path(__file__).resolve().parent.parent / "rag-api")
if _RAG_API not in sys.path:
    sys.path.insert(0, _RAG_API)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from v3_routes import router, RunMode


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_OK_RESULT = {
    "task_id": "test-1",
    "passed": True,
    "code": "print('hello')",
    "phase_solved": "phase1",
    "candidates_generated": 3,
    "total_tokens": 1500,
    "total_time_ms": 2500.0,
    "telemetry": {"adaptive_k": 3},
}


class TestV3RunEndpoint:

    @patch("v3_routes._run_v3_task", return_value=_OK_RESULT)
    def test_fast_mode_success(self, mock_run, client):
        resp = client.post("/v3/run", json={
            "task_id": "test-1",
            "prompt": "Solve two-sum",
            "mode": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "test-1"
        assert data["status"] == "solved"
        assert data["phase_solved"] == "phase1"
        assert data["candidates_generated"] == 3
        assert data["total_tokens"] == 1500

    @patch("v3_routes._run_v3_task")
    def test_thorough_mode_success(self, mock_run, client):
        captured = []
        mock_run.side_effect = lambda r: captured.append(r) or {
            **_OK_RESULT, "phase_solved": "refinement",
        }
        resp = client.post("/v3/run", json={
            "task_id": "test-2",
            "prompt": "Solve two-sum",
            "mode": "thorough",
            "test_code": "assert solution(1) == 2",
        })
        assert resp.status_code == 200
        assert resp.json()["phase_solved"] == "refinement"
        assert len(captured) == 1
        assert captured[0].mode == RunMode.THOROUGH

    def test_missing_prompt_returns_422(self, client):
        resp = client.post("/v3/run", json={"task_id": "t1"})
        assert resp.status_code == 422

    def test_missing_task_id_returns_422(self, client):
        resp = client.post("/v3/run", json={"prompt": "hello"})
        assert resp.status_code == 422

    def test_invalid_mode_returns_422(self, client):
        resp = client.post("/v3/run", json={
            "task_id": "t1", "prompt": "hello", "mode": "invalid",
        })
        assert resp.status_code == 422

    @patch("v3_routes._run_v3_task", return_value={
        **_OK_RESULT, "passed": False, "phase_solved": "none",
    })
    def test_failed_result_returns_status_failed(self, mock_run, client):
        resp = client.post("/v3/run", json={
            "task_id": "test-3", "prompt": "Impossible task",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    @patch("v3_routes._run_v3_task")
    def test_no_test_data_passes_none_fields(self, mock_run, client):
        captured = []
        mock_run.side_effect = lambda r: captured.append(r) or _OK_RESULT
        resp = client.post("/v3/run", json={
            "task_id": "test-4", "prompt": "Solve without tests",
        })
        assert resp.status_code == 200
        assert captured[0].test_code is None

    @patch("v3_routes._run_v3_task", side_effect=Exception("LLM down"))
    def test_pipeline_error_returns_500(self, mock_run, client):
        resp = client.post("/v3/run", json={
            "task_id": "test-5", "prompt": "Fail",
        })
        assert resp.status_code == 500
        assert "LLM down" in resp.json()["detail"]

    @patch("v3_routes._run_v3_task")
    def test_stdio_test_data_forwarded(self, mock_run, client):
        captured = []
        mock_run.side_effect = lambda r: captured.append(r) or _OK_RESULT
        resp = client.post("/v3/run", json={
            "task_id": "test-6",
            "prompt": "Add two numbers",
            "test_inputs": ["1 2", "3 4"],
            "test_outputs": ["3", "7"],
        })
        assert resp.status_code == 200
        assert captured[0].test_inputs == ["1 2", "3 4"]

    def test_timeout_below_minimum_returns_422(self, client):
        resp = client.post("/v3/run", json={
            "task_id": "t1", "prompt": "hello", "timeout_seconds": 0,
        })
        assert resp.status_code == 422

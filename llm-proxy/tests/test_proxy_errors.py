"""Tests for proxy_all error handling in llm-proxy (pardot-j3m.1).

Verifies that httpx exceptions return structured JSON error responses
instead of raw 500 Internal Server Error.
"""
import os
import sys

# Configure env before importing app
os.environ.setdefault("ALLOW_INTERNAL", "true")
os.environ.setdefault("LLAMA_URL", "http://fake-llama:8000")
os.environ.setdefault("REDIS_URL", "redis://fake-redis:6379")
os.environ.setdefault("API_PORTAL_URL", "http://fake-portal:3000")

# Ensure llm-proxy source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, patch

try:
    import httpx
    from main import app
    from fastapi.testclient import TestClient
    HAS_DEPS = True
except ImportError:
    httpx = None
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="fastapi/httpx/redis not installed")


def _make_client():
    """Create a TestClient that does NOT raise server exceptions."""
    return TestClient(app, raise_server_exceptions=False)


def _mock_async_client(side_effect):
    """Create a patched httpx.AsyncClient context manager that raises side_effect."""
    patcher = patch("main.httpx.AsyncClient")
    mock_cls = patcher.start()
    instance = AsyncMock()
    instance.get.side_effect = side_effect
    instance.request.side_effect = side_effect
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return patcher


class TestProxyAllReadTimeout:
    """proxy_all returns 504 with structured JSON on ReadTimeout."""

    def test_get_request_timeout(self):
        client = _make_client()
        patcher = _mock_async_client(httpx.ReadTimeout("timed out"))
        try:
            resp = client.get("/some/endpoint")
            assert resp.status_code == 504
            body = resp.json()
            assert body["error"]["type"] == "timeout_error"
            assert body["error"]["code"] == "read_timeout"
            assert "some/endpoint" in body["error"]["message"]
            assert body["error"]["param"] == "some/endpoint"
        finally:
            patcher.stop()

    def test_post_request_timeout(self):
        client = _make_client()
        patcher = _mock_async_client(httpx.ReadTimeout("timed out"))
        try:
            resp = client.post("/v1/embeddings", json={"input": "test"})
            assert resp.status_code == 504
            body = resp.json()
            assert body["error"]["code"] == "read_timeout"
        finally:
            patcher.stop()


class TestProxyAllConnectError:
    """proxy_all returns 502 with structured JSON on ConnectError."""

    def test_connect_error(self):
        client = _make_client()
        patcher = _mock_async_client(httpx.ConnectError("connection refused"))
        try:
            resp = client.get("/v1/slots")
            assert resp.status_code == 502
            body = resp.json()
            assert body["error"]["type"] == "backend_error"
            assert body["error"]["code"] == "connect_error"
            assert "v1/slots" in body["error"]["message"]
        finally:
            patcher.stop()


class TestProxyAllGenericHTTPError:
    """proxy_all returns 502 with structured JSON on other httpx errors."""

    def test_write_timeout(self):
        client = _make_client()
        patcher = _mock_async_client(httpx.WriteTimeout("write timed out"))
        try:
            resp = client.post("/v1/tokenize", json={"content": "test"})
            assert resp.status_code == 502
            body = resp.json()
            assert body["error"]["type"] == "backend_error"
            assert body["error"]["code"] == "proxy_error"
            assert "WriteTimeout" in body["error"]["message"]
        finally:
            patcher.stop()


class TestProxyAllRateLimitHeadersOnError:
    """Rate limit headers are included even in error responses."""

    def test_timeout_includes_rate_headers(self):
        client = _make_client()
        patcher = _mock_async_client(httpx.ReadTimeout("timed out"))
        try:
            resp = client.get("/some/path")
            assert resp.status_code == 504
            assert "x-ratelimit-limit" in resp.headers
            assert "x-ratelimit-remaining" in resp.headers
        finally:
            patcher.stop()

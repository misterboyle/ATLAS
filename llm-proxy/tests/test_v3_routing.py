"""Tests for atlas-v3 model routing in llm-proxy."""

import json
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure main module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("ALLOW_INTERNAL", "true")

_mock_redis = MagicMock()
_mock_redis.ping.return_value = True
_mock_redis.incr.return_value = 1
_mock_redis.expire.return_value = True
_mock_redis.ttl.return_value = 60

with patch("redis.from_url", return_value=_mock_redis):
    from main import app, ATLAS_V3_MODELS

from fastapi.testclient import TestClient

client = TestClient(app)


def _rag_ok(answer="Test answer", **extra):
    data = {"answer": answer}
    data.update(extra)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _backend_models():
    data = {"object": "list", "data": [
        {"id": "local-model", "object": "model",
         "created": 1700000000, "owned_by": "local"}
    ]}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    return resp


def _patch_client(mock_resp):
    """Return (patcher, mock_instance) for httpx.AsyncClient."""
    mi = AsyncMock()
    mi.post.return_value = mock_resp
    mi.get.return_value = mock_resp
    p = patch("main.httpx.AsyncClient")
    mc = p.start()
    mc.return_value.__aenter__.return_value = mi
    return p, mi


class TestV3Routing:
    def test_v3_fast_routes_to_rag(self):
        p, mi = _patch_client(_rag_ok("Fast result"))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-fast",
                "messages": [{"role": "user", "content": "What is ATLAS?"}]
            })
            assert r.status_code == 200
            d = r.json()
            assert d["model"] == "atlas-v3-fast"
            assert d["choices"][0]["message"]["content"] == "Fast result"
            assert d["v3_metadata"]["mode"] == "fast"
            assert mi.post.called
            call_url = str(mi.post.call_args)
            assert "/v3/run" in call_url
        finally:
            p.stop()

    def test_v3_thorough_mode(self):
        p, mi = _patch_client(_rag_ok("Thorough result"))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-thorough",
                "messages": [{"role": "user", "content": "Explain X"}]
            })
            assert r.status_code == 200
            assert r.json()["v3_metadata"]["mode"] == "thorough"
            sent = mi.post.call_args.kwargs.get("json", {})
            assert sent.get("mode") == "thorough"
        finally:
            p.stop()

    def test_v3_bare_defaults_fast(self):
        p, _ = _patch_client(_rag_ok("Default"))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3",
                "messages": [{"role": "user", "content": "Hi"}]
            })
            assert r.status_code == 200
            assert r.json()["v3_metadata"]["mode"] == "fast"
        finally:
            p.stop()

    def test_prompt_from_last_user_message(self):
        p, mi = _patch_client(_rag_ok("Answer"))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-fast",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "First Q"},
                    {"role": "assistant", "content": "First A"},
                    {"role": "user", "content": "Second Q"},
                ]
            })
            assert r.status_code == 200
            sent = mi.post.call_args.kwargs.get("json", {})
            assert sent["prompt"] == "Second Q"
        finally:
            p.stop()

    def test_openai_response_format(self):
        p, _ = _patch_client(_rag_ok(
            "Formatted", sources=["doc1"], confidence=0.95
        ))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-fast",
                "messages": [{"role": "user", "content": "Test"}]
            })
            d = r.json()
            assert d["id"].startswith("chatcmpl-v3-")
            assert d["object"] == "chat.completion"
            assert "created" in d
            assert len(d["choices"]) == 1
            assert d["choices"][0]["finish_reason"] == "stop"
            assert d["choices"][0]["message"]["role"] == "assistant"
            u = d["usage"]
            assert "prompt_tokens" in u
            assert "completion_tokens" in u
            assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
            assert d["v3_metadata"]["sources"] == ["doc1"]
            assert d["v3_metadata"]["confidence"] == 0.95
        finally:
            p.stop()

    def test_rag_connect_error_returns_502(self):
        import httpx as httpx_mod
        mi = AsyncMock()
        mi.post.side_effect = httpx_mod.ConnectError("refused")
        p = patch("main.httpx.AsyncClient")
        mc = p.start()
        mc.return_value.__aenter__.return_value = mi
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-fast",
                "messages": [{"role": "user", "content": "Test"}]
            })
            assert r.status_code == 502
            assert r.json()["error"]["code"] == "connect_error"
        finally:
            p.stop()

    def test_no_user_message_returns_400(self):
        p, _ = _patch_client(_rag_ok("X"))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "atlas-v3-fast",
                "messages": [{"role": "system", "content": "Only system"}]
            })
            assert r.status_code in (400, 422)
        finally:
            p.stop()

    def test_v3_models_in_list(self):
        p, _ = _patch_client(_backend_models())
        try:
            r = client.get("/v1/models")
            assert r.status_code == 200
            ids = [m["id"] for m in r.json()["data"]]
            assert "atlas-v3" in ids
            assert "atlas-v3-fast" in ids
            assert "atlas-v3-thorough" in ids
            assert "local-model" in ids
        finally:
            p.stop()

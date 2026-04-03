"""Tests for V3 Pipeline API endpoints."""
import os
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "rag-api"))
sys.path.insert(0, os.path.join(_here, ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from v3_routes import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


def test_plan_search_configured():
    r = client.post("/v3/plan-search", json={"task_id": "t1", "problem": "Sort"})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "configured"
    assert d["config"]["max_plans"] == 7


def test_plan_search_parse():
    text = ("CONSTRAINT SET 1:\n- Constraint: O(n log n)\n"
            "- Eliminates: brute\n- Implies: sorting\n"
            "CONSTRAINT SET 2:\n- Constraint: O(1) space\n"
            "- Eliminates: hash\n- Implies: in-place\n")
    r = client.post("/v3/plan-search", json={
        "task_id": "t2", "problem": "Sort", "constraint_response": text})
    assert r.status_code == 200
    assert r.json()["status"] == "parsed"
    assert len(r.json()["constraint_sets"]) == 2


def test_budget_forcing_with_energy():
    r = client.post("/v3/budget-forcing", json={"task_id": "t3", "raw_energy": 15.0})
    assert r.status_code == 200
    d = r.json()
    assert d["tier"] in ("nothink", "standard", "hard", "extreme")
    assert "max_thinking" in d["tier_config"]
    assert d["normalized_energy"] is not None


def test_budget_forcing_default():
    r = client.post("/v3/budget-forcing", json={
        "task_id": "t4", "default_tier": "hard"})
    assert r.status_code == 200
    assert r.json()["tier"] == "hard"


def test_budget_forcing_thinking():
    r = client.post("/v3/budget-forcing", json={
        "task_id": "t5",
        "response_text": "<think>Consider edge cases.</think>def solve(): pass"})
    assert r.status_code == 200
    d = r.json()
    assert d["thinking"] is not None
    assert d["thinking"]["thinking_length"] > 0
    assert d["thinking"]["output_length"] > 0


def test_pr_cot_configured():
    r = client.post("/v3/pr-cot", json={
        "task_id": "t6", "problem": "Sum",
        "code": "def f(): return 0", "error": "WA"})
    assert r.status_code == 200
    d = r.json()
    assert len(d["perspectives"]) == 4
    assert "logical_consistency" in d["perspectives"]


def test_pr_cot_extract():
    repair = "Fix:\n```python\ndef f():\n    return 42\n```"
    r = client.post("/v3/pr-cot", json={
        "task_id": "t7", "problem": "42", "code": "def f(): return 0",
        "repair_response": repair})
    assert r.status_code == 200
    assert "42" in r.json()["extracted_code"]


def test_self_test_gen_configured():
    r = client.post("/v3/self-test-gen", json={
        "task_id": "t8", "problem": "Add"})
    assert r.status_code == 200
    assert r.json()["config"]["num_test_cases"] == 5


def test_self_test_gen_parse():
    text = ("TEST CASE 1:\nDESCRIPTION: Simple\nINPUT: 2 3\nOUTPUT: 5\n"
            "TEST CASE 2:\nDESCRIPTION: Zero\nINPUT: 0 0\nOUTPUT: 0\n")
    r = client.post("/v3/self-test-gen", json={
        "task_id": "t9", "problem": "Add", "generation_response": text})
    assert r.status_code == 200
    assert len(r.json()["test_cases"]) == 2


def test_sandbox_execute():
    r = client.post("/v3/sandbox", json={
        "task_id": "t10", "code": "print('hello')"})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "success"
    assert "hello" in d["stdout"]
    assert d["exit_code"] == 0


def test_sandbox_stdin():
    r = client.post("/v3/sandbox", json={
        "task_id": "t11",
        "code": "x=input()\nprint(f'got:{x}')", "stdin": "test"})
    assert r.status_code == 200
    assert "got:test" in r.json()["stdout"]


def test_sandbox_empty():
    r = client.post("/v3/sandbox", json={"task_id": "t12", "code": "   "})
    assert r.status_code == 400

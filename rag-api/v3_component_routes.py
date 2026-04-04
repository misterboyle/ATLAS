"""V3 Component debug/inspection endpoints."""
import os
import re
import subprocess
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_ATLAS_ROOT = os.environ.get("ATLAS_ROOT", os.path.join(os.path.dirname(__file__), ".."))
if _ATLAS_ROOT not in sys.path:
    sys.path.insert(0, _ATLAS_ROOT)

router = APIRouter()


# --- Request models ---

class PlanSearchReq(BaseModel):
    task_id: str
    problem: str
    constraint_response: Optional[str] = None

class BudgetForcingReq(BaseModel):
    task_id: str
    raw_energy: Optional[float] = None
    default_tier: Optional[str] = None
    response_text: Optional[str] = None

class PRCoTReq(BaseModel):
    task_id: str
    problem: str
    code: str = ""
    error: str = ""
    repair_response: Optional[str] = None

class SelfTestGenReq(BaseModel):
    task_id: str
    problem: str
    generation_response: Optional[str] = None

class SandboxReq(BaseModel):
    task_id: str
    code: str
    stdin: Optional[str] = None


# --- Endpoints ---

@router.post("/plan-search")
def plan_search(req: PlanSearchReq):
    max_plans = int(os.environ.get("ATLAS_V3_PLAN_SEARCH_MAX_PLANS", "7"))
    if req.constraint_response:
        sets = []
        current = None
        for line in req.constraint_response.strip().splitlines():
            m = re.match(r"CONSTRAINT SET (\d+):", line)
            if m:
                if current:
                    sets.append(current)
                current = {"id": int(m.group(1)), "constraints": [], "eliminates": [], "implies": []}
            elif current:
                cm = re.match(r"- Constraint:\s*(.+)", line)
                em = re.match(r"- Eliminates:\s*(.+)", line)
                im = re.match(r"- Implies:\s*(.+)", line)
                if cm:
                    current["constraints"].append(cm.group(1))
                elif em:
                    current["eliminates"].append(em.group(1))
                elif im:
                    current["implies"].append(im.group(1))
        if current:
            sets.append(current)
        return {"status": "parsed", "constraint_sets": sets}
    return {"status": "configured", "config": {"max_plans": max_plans}}


@router.post("/budget-forcing")
def budget_forcing(req: BudgetForcingReq):
    try:
        from benchmark.v3.budget_forcing import BudgetForcing, BudgetForcingConfig, BUDGET_TIERS
    except ImportError:
        raise HTTPException(500, "BudgetForcing not available")
    bf = BudgetForcing(BudgetForcingConfig(enabled=True))
    if req.response_text:
        think_m = re.search(r"<think>(.*?)</think>", req.response_text, re.DOTALL)
        thinking_text = think_m.group(1) if think_m else ""
        output_text = re.sub(r"<think>.*?</think>", "", req.response_text, flags=re.DOTALL)
        return {"tier": "standard", "tier_config": bf.get_tier_config("standard"),
                "thinking": {"thinking_length": len(thinking_text),
                             "output_length": len(output_text.strip())},
                "normalized_energy": None}
    tier = req.default_tier or "standard"
    norm_e = None
    if req.raw_energy is not None:
        norm_e = 1.0 / (1.0 + abs(req.raw_energy))
        if req.raw_energy < 5:
            tier = "nothink"
        elif req.raw_energy < 10:
            tier = "standard"
        elif req.raw_energy < 20:
            tier = "hard"
        else:
            tier = "extreme"
    tc = bf.get_tier_config(tier)
    return {"tier": tier, "tier_config": tc, "normalized_energy": norm_e}


@router.post("/pr-cot")
def pr_cot(req: PRCoTReq):
    if req.repair_response:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", req.repair_response, re.DOTALL)
        code = m.group(1).strip() if m else req.repair_response
        return {"status": "extracted", "extracted_code": code}
    perspectives = {
        "logical_consistency": "Check logical correctness",
        "edge_cases": "Consider edge cases",
        "efficiency": "Evaluate time/space complexity",
        "specification_adherence": "Verify meets problem spec",
    }
    return {"status": "configured", "perspectives": perspectives}


@router.post("/self-test-gen")
def self_test_gen(req: SelfTestGenReq):
    num_cases = int(os.environ.get("ATLAS_V3_SELF_TEST_NUM_CASES", "5"))
    if req.generation_response:
        cases = []
        current = None
        for line in req.generation_response.strip().splitlines():
            tm = re.match(r"TEST CASE (\d+):", line)
            if tm:
                if current:
                    cases.append(current)
                current = {"description": "", "input_str": "", "expected_output": ""}
            elif current:
                dm = re.match(r"DESCRIPTION:\s*(.+)", line)
                im = re.match(r"INPUT:\s*(.+)", line)
                om = re.match(r"OUTPUT:\s*(.+)", line)
                if dm:
                    current["description"] = dm.group(1)
                elif im:
                    current["input_str"] = im.group(1)
                elif om:
                    current["expected_output"] = om.group(1)
        if current:
            cases.append(current)
        return {"status": "parsed", "test_cases": cases}
    return {"status": "configured", "config": {"num_test_cases": num_cases}}


@router.post("/sandbox")
def sandbox(req: SandboxReq):
    if not req.code.strip():
        raise HTTPException(400, "Empty code")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", req.code],
            input=req.stdin, capture_output=True, text=True, timeout=30,
        )
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "TLE", "exit_code": -1}
    except Exception as e:
        raise HTTPException(500, str(e))

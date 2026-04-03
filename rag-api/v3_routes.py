"""V3 Pipeline API endpoints -- REST facade for benchmark/v3/ modules."""
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v3", tags=["v3-pipeline"])

_bp = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _bp not in sys.path:
    sys.path.insert(0, _bp)


# -- Request / Response models ----------------------------------------------

class PlanSearchRequest(BaseModel):
    task_id: str
    problem: str
    max_plans: int = Field(default=7, ge=1, le=20)
    step1_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    step2_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    step3_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    constraint_response: Optional[str] = None


class PlanSearchResponse(BaseModel):
    task_id: str
    status: str
    config: Dict
    constraint_sets: List[Dict] = []
    message: str = ""


class BudgetForcingRequest(BaseModel):
    task_id: str
    raw_energy: Optional[float] = None
    normalized_energy: Optional[float] = None
    response_text: Optional[str] = None
    default_tier: str = Field(default="standard")


class BudgetForcingResponse(BaseModel):
    task_id: str
    tier: str
    tier_config: Dict
    system_prompt: str
    max_tokens: int
    normalized_energy: Optional[float] = None
    thinking: Optional[Dict] = None


class PRCoTRequest(BaseModel):
    task_id: str
    problem: str
    code: str
    error: str = ""
    max_repair_rounds: int = Field(default=3, ge=1, le=10)
    repair_response: Optional[str] = None


class PRCoTResponse(BaseModel):
    task_id: str
    status: str
    config: Dict
    perspectives: List[str]
    extracted_code: Optional[str] = None
    message: str = ""


class SelfTestGenRequest(BaseModel):
    task_id: str
    problem: str
    num_test_cases: int = Field(default=5, ge=1, le=20)
    generation_response: Optional[str] = None


class SelfTestGenResponse(BaseModel):
    task_id: str
    status: str
    config: Dict
    test_cases: List[Dict] = []
    message: str = ""


class SandboxRequest(BaseModel):
    task_id: str
    code: str
    stdin: str = ""
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class SandboxResponse(BaseModel):
    task_id: str
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False


# -- Endpoints --------------------------------------------------------------

@router.post("/plan-search", response_model=PlanSearchResponse)
async def v3_plan_search(request: PlanSearchRequest):
    """Configure PlanSearch and optionally parse constraint responses."""
    try:
        from benchmark.v3.plan_search import PlanSearchConfig, parse_constraint_sets
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    cfg = PlanSearchConfig(
        enabled=True, max_plans=request.max_plans,
        step1_temperature=request.step1_temperature,
        step2_temperature=request.step2_temperature,
        step3_temperature=request.step3_temperature,
    )
    cs = []
    if request.constraint_response:
        parsed = parse_constraint_sets(request.constraint_response, cfg.max_plans)
        cs = [c.to_dict() for c in parsed]
    return PlanSearchResponse(
        task_id=request.task_id,
        status="parsed" if cs else "configured",
        config={"enabled": True, "max_plans": cfg.max_plans,
                "step1_temperature": cfg.step1_temperature,
                "step2_temperature": cfg.step2_temperature,
                "step3_temperature": cfg.step3_temperature},
        constraint_sets=cs,
        message=f"Parsed {len(cs)} constraint sets" if cs
                else "PlanSearch configured; supply constraint_response to parse",
    )


@router.post("/budget-forcing", response_model=BudgetForcingResponse)
async def v3_budget_forcing(request: BudgetForcingRequest):
    """Select budget tier and optionally analyze thinking in a response."""
    try:
        from benchmark.v3.budget_forcing import (
            BudgetForcing, BudgetForcingConfig, normalize_energy,
            get_system_prompt, extract_thinking, estimate_thinking_tokens,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    cfg = BudgetForcingConfig(enabled=True, default_tier=request.default_tier)
    bf = BudgetForcing(cfg)
    tier = bf.select_tier(raw_energy=request.raw_energy,
                          normalized_energy=request.normalized_energy)
    norm = None
    if request.raw_energy is not None:
        norm = normalize_energy(request.raw_energy)
    elif request.normalized_energy is not None:
        norm = request.normalized_energy
    thinking = None
    if request.response_text:
        tt, ot = extract_thinking(request.response_text)
        thinking = {
            "thinking_text": tt[:500], "output_preview": ot[:500],
            "estimated_tokens": estimate_thinking_tokens(tt),
            "thinking_length": len(tt), "output_length": len(ot),
        }
    return BudgetForcingResponse(
        task_id=request.task_id, tier=tier,
        tier_config=bf.get_tier_config(tier),
        system_prompt=get_system_prompt(tier),
        max_tokens=bf.get_max_tokens(tier),
        normalized_energy=norm, thinking=thinking,
    )


@router.post("/pr-cot", response_model=PRCoTResponse)
async def v3_pr_cot(request: PRCoTRequest):
    """Configure PR-CoT repair and optionally extract code from a response."""
    try:
        from benchmark.v3.pr_cot import (
            PRCoTConfig, PERSPECTIVES, extract_code_from_repair,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    cfg = PRCoTConfig(enabled=True, max_repair_rounds=request.max_repair_rounds)
    extracted = None
    if request.repair_response:
        extracted = extract_code_from_repair(request.repair_response)
    return PRCoTResponse(
        task_id=request.task_id,
        status="parsed" if extracted else "configured",
        config={"enabled": True, "max_repair_rounds": cfg.max_repair_rounds,
                "analysis_temperature": cfg.analysis_temperature,
                "repair_temperature": cfg.repair_temperature},
        perspectives=list(PERSPECTIVES.keys()),
        extracted_code=extracted,
        message="Extracted repair code" if extracted
                else "PR-CoT configured with 4 perspectives",
    )


@router.post("/self-test-gen", response_model=SelfTestGenResponse)
async def v3_self_test_gen(request: SelfTestGenRequest):
    """Configure SelfTestGen and optionally parse test cases from a response."""
    try:
        from benchmark.v3.self_test_gen import SelfTestGenConfig, SelfTestGen
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    cfg = SelfTestGenConfig(enabled=True, num_test_cases=request.num_test_cases)
    cases = []
    if request.generation_response:
        gen = SelfTestGen(cfg)
        parsed = gen.parse_test_cases(request.generation_response)
        cases = [tc.to_dict() for tc in parsed]
    return SelfTestGenResponse(
        task_id=request.task_id,
        status="parsed" if cases else "configured",
        config={"enabled": True, "num_test_cases": cfg.num_test_cases,
                "generation_temperature": cfg.generation_temperature,
                "majority_threshold": cfg.majority_threshold},
        test_cases=cases,
        message=f"Parsed {len(cases)} test cases" if cases
                else "SelfTestGen configured; supply generation_response to parse",
    )


@router.post("/sandbox", response_model=SandboxResponse)
async def v3_sandbox(request: SandboxRequest):
    """Execute Python code in an isolated subprocess with timeout."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(request.code)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path], input=request.stdin,
            capture_output=True, text=True, timeout=request.timeout_seconds,
        )
        return SandboxResponse(
            task_id=request.task_id,
            status="success" if result.returncode == 0 else "error",
            stdout=result.stdout[:10000], stderr=result.stderr[:10000],
            exit_code=result.returncode, timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResponse(
            task_id=request.task_id, status="timeout",
            stderr=f"Exceeded {request.timeout_seconds}s limit", timed_out=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

"""V3 Pipeline service endpoint -- POST /v3/run wrapping V3Pipeline.run_task()."""

import asyncio
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_ATLAS_ROOT = Path(__file__).resolve().parent.parent
if str(_ATLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ATLAS_ROOT))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v3", tags=["v3-pipeline"])

_executor = ThreadPoolExecutor(max_workers=2)


class RunMode(str, Enum):
    FAST = "fast"
    THOROUGH = "thorough"
    AUTO = "auto"


class V3RunRequest(BaseModel):
    task_id: str
    prompt: str
    mode: RunMode = RunMode.FAST
    test_code: Optional[str] = None
    test_inputs: Optional[List[str]] = None
    test_outputs: Optional[List[str]] = None
    timeout_seconds: int = Field(default=300, ge=1, le=3600)


class V3RunResponse(BaseModel):
    task_id: str
    status: str
    code: str
    phase_solved: str
    candidates_generated: int
    total_tokens: int
    total_time_ms: float
    telemetry: Dict[str, Any]


def _run_v3_task(req: V3RunRequest) -> Dict[str, Any]:
    """Synchronous entry point executed in the thread pool."""
    from benchmark.models import BenchmarkTask
    from benchmark.runner import BenchmarkRunner
    from benchmark.v3_runner import V3Pipeline, LLMAdapter
    from benchmark.v3.self_test_gen import SelfTestGen, SelfTestGenConfig

    v3_llama_url = os.environ.get("V3_LLAMA_URL", "http://localhost:32735")
    os.environ.setdefault("LLAMA_URL", v3_llama_url)

    has_stdio = bool(req.test_inputs and req.test_outputs)
    has_functional = bool(req.test_code)
    if has_stdio:
        eval_mode = "stdio"
    elif has_functional:
        eval_mode = "functional"
    else:
        eval_mode = "stdio"

    task = BenchmarkTask(
        task_id=req.task_id,
        canonical_solution="",
        entry_point="solution",
        prompt=req.prompt,
        eval_mode=eval_mode,
        test_code=req.test_code or "",
        test_inputs=req.test_inputs or [],
        test_outputs=req.test_outputs or [],
    )

    telemetry_dir = Path(tempfile.mkdtemp(prefix="v3_api_"))
    runner = BenchmarkRunner(max_retries=4)
    runner.llm_url = v3_llama_url

    try:
        enable_phase3 = req.mode in (RunMode.THOROUGH, RunMode.AUTO)
        auto_phase3 = req.mode == RunMode.AUTO

        pipeline = V3Pipeline(
            runner=runner,
            telemetry_dir=telemetry_dir,
            llama_url=v3_llama_url,
            enable_phase1=True,
            enable_phase2=True,
            enable_phase3=enable_phase3,
            auto_phase3=auto_phase3,
        )

        if not has_stdio and not has_functional:
            llm = LLMAdapter(runner)
            self_test_gen = SelfTestGen(
                SelfTestGenConfig(enabled=True),
                telemetry_dir=telemetry_dir,
            )
            st_result = self_test_gen.generate(
                problem=req.prompt, llm_call=llm, task_id=req.task_id,
            )
            if st_result.test_cases:
                task.test_inputs = [
                    tc.input_str for tc in st_result.test_cases
                ]
                task.test_outputs = [
                    tc.expected_output for tc in st_result.test_cases
                ]
                logger.info(
                    "SelfTestGen produced %d tests for task %s",
                    len(st_result.test_cases), req.task_id,
                )

        return pipeline.run_task(task, task_id=req.task_id)
    finally:
        runner.close()


@router.post("/run", response_model=V3RunResponse)
async def v3_run(request: V3RunRequest):
    """Run a task through the V3 pipeline.

    mode=fast:     Phase 1 only (generate + test candidates).
    mode=thorough: Phase 1 + Phase 3 (refinement cascade).
    """
    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_v3_task, request),
            timeout=request.timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Task timed out after {request.timeout_seconds}s",
        )
    except Exception as e:
        logger.exception("V3 pipeline error for task %s", request.task_id)
        raise HTTPException(status_code=500, detail=str(e))

    status = "solved" if result.get("passed") else "failed"

    return V3RunResponse(
        task_id=result.get("task_id", request.task_id),
        status=status,
        code=result.get("code", ""),
        phase_solved=result.get("phase_solved", "none"),
        candidates_generated=result.get("candidates_generated", 0),
        total_tokens=result.get("total_tokens", 0),
        total_time_ms=result.get("total_time_ms", 0.0),
        telemetry=result.get("telemetry", {}),
    )

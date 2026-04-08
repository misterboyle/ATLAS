# Benchmark Infrastructure and Results

## Results (Qwen3-14B-Q4_K_M, RTX 5060 Ti 16GB)

| Benchmark | Score | Tasks | Pipeline |
|-----------|-------|-------|----------|
| LiveCodeBench v5 | 74.6% pass@1-v(k=3) | 599 | V3 |
| GPQA Diamond | 47.0% | 198 | V2 |
| SciCode | 14.7% (sub-problems) | 341 | V2 |

pass@1-v(k=3) = one solution submitted per task, generated via
best-of-3 candidates + Lens selection + iterative repair.

## V3 Ablation (LiveCodeBench)

| Condition | Config | Pass Rate | Delta |
|-----------|--------|-----------|-------|
| A | Baseline (no V3) | 54.9% | -- |
| B | +Phase 1 | 67.3% | +12.4pp |
| C | +Phase 1+2 (Lens) | 67.3% | +0.0pp |
| D | +Phase 1+3 (repair) | 74.6% | +7.3pp |

Data: `v3_ablation_results/` (5 conditions, 599 per-task JSON each)

## Dataset Loaders

All in `benchmark/datasets/`, download from HuggingFace (JSON rows API):

| Dataset | Tasks | Eval Mode | File |
|---------|-------|-----------|------|
| HumanEval | 164 | function | humaneval.py |
| MBPP | 500 | function | mbpp.py |
| HumanEval+ | 164 | function | evalplus_humaneval.py |
| MBPP+ | 500 | function | evalplus_mbpp.py |
| LiveCodeBench v5 | 599 | stdio | livecodebench.py |
| GPQA Diamond | 198 | mcq | gpqa.py |
| IFBench | 300 | ifbench | ifbench.py |
| SciCode | ~80 | function | scicode.py |

## Runner Infrastructure

- `benchmark/runner.py` -- Core execution, LLM API, code extraction
- `benchmark/v3_runner.py` -- V3 runner with ablation conditions A-F
- `benchmark/v2_runner.py` -- V2 runner (phases 0-6, telemetry)
- `benchmark/cli.py` -- CLI: `atlas benchmark --humaneval --dry-run`
- `benchmark/config.py` -- BenchmarkConfig from atlas.conf
- `benchmark/models.py` -- BenchmarkTask, AttemptResult, TaskResult

## Analysis

- `benchmark/analysis/pass_at_k.py` -- pass@k metric
- `benchmark/analysis/cost_analysis.py` -- Token/electricity costs
- `benchmark/analysis/hardware_info.py` -- GPU/CPU detection

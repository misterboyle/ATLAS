# V3 Pipeline (Inner Layer)

Activates for T2+ files via write_file/edit_file. Four phases with
early exits at every stage.

## Phase 0: Probe

Single baseline candidate with progressive retry:
light (1024 thinking) -> standard (2048) -> /nothink (0).
Scored with C(x)/G(x) and tested in sandbox. If it passes, exit.

## Phase 1: Constraint-Driven Generation

- **PlanSearch (1A):** 3 structurally different implementation plans
- **DivSampling (1B):** 12 perturbations = 4 roles + 4 instructions +
  4 styles
  - Roles: competitive_programmer, systems_engineer, mathematician,
    pragmatist
  - Instructions: step_by_step, edge_case_first, complexity_aware,
    constraint_driven
  - Styles: functional, pythonic, optimize_iteratively, structured
- **BudgetForcing (1C):** 5 thinking tiers with Wait injection
  - nothink (0), light (1024), standard (2048), hard (4096),
    extreme (8192)
  - Wait injection: "Wait, let me reconsider." forces longer thinking

## Phase 2: Verification and Selection

- **BlendASC (2A):** Adaptive K allocation from C(x) energy
- **Build Verification:** Per-language syntax checks (py_compile, tsc,
  go build, cargo check, gcc, bash -n)
- **S* Tiebreaking (2C):** When 2+ candidates pass, generate edge-case
  inputs, run both, majority wins
- **Lens Selection:** When 1 passes or fallback, sort by C(x) energy,
  lowest wins

## Phase 3: Repair (0/K pass)

Three strategies, sequential with early exit:

1. **Failure Analysis (3A):** Categorize into 6 types (wrong_algorithm,
   implementation_bug, edge_case_miss, time_limit, format_error,
   partial_correct)
2. **Metacognitive (3F):** Inject compensating constraints from known
   Qwen failure patterns
3. **PR-CoT (3C):** 4 perspectives x (analysis + repair) = ~8 LLM calls,
   up to 3 rounds
4. **Refinement Loop (3E):** Failure Analysis -> Constraint Refinement ->
   Code Gen -> Test -> Learn. 2 iterations, 120s budget, cosine
   distance >= 0.15 prevents hypothesis repetition
5. **Derivation Chains (3D):** Decompose into up to 5 sub-problems,
   sandbox-verify each, compose final. ~7+ LLM calls

## 19 Pipeline Modules

All in `benchmark/v3/`, orchestrated by `v3-service/main.py`:

| Module | Phase | Purpose |
|--------|-------|---------|
| plan_search.py | 1A | 3 constraint-based plans |
| div_sampling.py | 1B | 12 perturbations |
| budget_forcing.py | 1C | 5 tiers, Wait injection |
| blend_asc.py | 2A | Adaptive K allocation |
| reasc.py | 2B | Early stopping |
| s_star.py | 2C | Differential tiebreaking |
| candidate_selection.py | 2 | 4 selection strategies |
| failure_analysis.py | 3A | 6 failure categories |
| constraint_refinement.py | 3B | Cosine filtering |
| pr_cot.py | 3C | 4-perspective repair |
| derivation_chains.py | 3D | Sub-problem decomposition |
| refinement_loop.py | 3E | Orchestrator |
| metacognitive.py | 3F | Failure pattern library |
| ace_pipeline.py | 3G | Playbook learning |
| self_test_gen.py | util | Model-generated tests |
| lens_feedback.py | util | Online recalibration |
| embedding_store.py | util | Binary persistence |
| ablation_analysis.py | util | Statistical analysis |

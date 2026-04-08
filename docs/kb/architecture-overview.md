# ATLAS Architecture Overview

ATLAS V3.0.1 uses a **two-layer design**:

1. **Outer Layer (Agent Loop):** `atlas-proxy` (Go) receives chat requests
   from Aider, runs a grammar-constrained agent loop with 8 tools, and
   routes complex files through the V3 pipeline.
2. **Inner Layer (V3 Pipeline):** `v3-service` (Python) generates diverse
   code candidates with build verification and energy-based selection.

## 5-Service Stack

| Service | Port | Language | Role |
|---------|------|----------|------|
| llama-server | 8080 | C++ (llama.cpp) | LLM inference (GPU) |
| atlas-proxy | 8090 | Go | Agent loop, tool routing |
| v3-service | 8070 | Python | V3 pipeline orchestrator |
| geometric-lens | 8099 | Python (FastAPI) | Scoring, RAG, routing |
| sandbox | 30820 | Python (FastAPI) | Isolated code execution |

Only `llama-server` uses the GPU. Everything else runs on CPU.

## Data Flow Summary

- **T1 (simple files):** Aider -> proxy -> LLM -> direct write. One LLM call.
- **T2+ (complex files):** Aider -> proxy -> LLM -> V3 pipeline ->
  (PlanSearch + DivSampling + scoring + sandbox) -> best candidate.
- **Scoring:** V3 calls geometric-lens for C(x)/G(x) energy scoring.
- **Verification:** V3 calls sandbox for build checks and test execution.

## Key Design Principles

- **Grammar-constrained output:** Every LLM response is forced into valid
  JSON via `response_format: json_object` at the llama-server level.
- **Early exit everywhere:** Pipeline exits as soon as a passing candidate
  is found (Phase 0 probe, Phase 1 generation, Phase 3 repair).
- **Frozen model:** No fine-tuning. All intelligence is in the pipeline
  infrastructure wrapping a standard Qwen model.
- **Single GPU:** RTX 5060 Ti 16GB. ~8.2 GB VRAM for llama-server,
  ~7.8 GB free headroom.

## Source References

- Architecture doc: `docs/ARCHITECTURE.md`
- Repository map: `docs/MAP.md`
- Service configs: `docker-compose.yml`

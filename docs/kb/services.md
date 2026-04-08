# ATLAS Services (5-Service Stack)

## llama-server (port 8080)

- **Language:** C++ (llama.cpp)
- **Model:** Qwen3.5-9B-Q6_K (V3.0.1) or Qwen3-14B-Q4_K_M (benchmarks)
- **Features:** CUDA inference, grammar-constrained JSON generation,
  self-embeddings via `/v1/embeddings`
- **VRAM:** ~8.2 GB (model weights + 32K KV cache)
- **Source:** `inference/` (Dockerfiles and entrypoint scripts)

## atlas-proxy (port 8090)

- **Language:** Go
- **Role:** Core agent loop. Receives OpenAI-compatible chat requests from
  Aider, classifies files into tiers (T0-T3), routes T2+ to V3 pipeline.
- **Features:** 8 tools, grammar enforcement, exploration budget,
  verify-repair loop, best-of-K, Aider format translation
- **Source:** `atlas-proxy/` (main.go, agent.go, tools.go, etc.)

## v3-service (port 8070)

- **Language:** Python
- **Role:** V3 pipeline HTTP wrapper. Orchestrates PlanSearch, DivSampling,
  BudgetForcing, PR-CoT, and all repair strategies.
- **Source:** `v3-service/main.py` (orchestrator),
  `benchmark/v3/` (19 pipeline modules)

## geometric-lens (port 8099)

- **Language:** Python (FastAPI)
- **Role:** Neural scoring (C(x) cost field, G(x) XGBoost), RAG/project
  indexing, confidence routing (Thompson Sampling), pattern cache
- **CPU-only:** ~12 MB for models, ~128 MB PyTorch runtime
- **Source:** `geometric-lens/` (main.py, geometric_lens/, indexer/,
  retriever/, router/, cache/)

## sandbox (port 30820 host / 8020 container)

- **Language:** Python (FastAPI)
- **Role:** Isolated code execution for 8 languages. Build verification,
  test running, linting, error classification.
- **Source:** `sandbox/executor_server.py`

## Startup Order

1. `llama-server` and `sandbox` start independently
2. `geometric-lens` and `v3-service` wait for llama-server healthy
3. `atlas-proxy` waits for all four services healthy

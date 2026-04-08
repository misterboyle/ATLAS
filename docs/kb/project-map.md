# Repository Structure and File Map

Top-level directories and their purposes.

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `atlas-proxy/` | Go proxy: agent loop, grammar, tools (12 .go files) |
| `atlas/` | Python CLI package (REPL, commands, client) |
| `benchmark/` | Benchmark runner, 9 dataset loaders, 19 V3 modules |
| `geometric-lens/` | Scoring (C(x)/G(x)), RAG, routing, pattern cache |
| `v3-service/` | V3 pipeline HTTP wrapper (main.py + Dockerfile) |
| `sandbox/` | Isolated code execution (executor_server.py) |
| `inference/` | llama-server Dockerfiles and entrypoint scripts |
| `scripts/` | Build, deploy, training automation (~20 scripts) |
| `tests/` | Test suite: infrastructure, integration, V3 unit |
| `docs/` | Architecture, API, CLI, setup, troubleshooting |
| `v3_ablation_results/` | Published ablation data (5 conditions) |

## Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | 5-service stack definition |
| `pyproject.toml` | Python package (atlas CLI entry point) |
| `.env.example` | Docker Compose environment template |
| `atlas.conf.example` | K3s deployment config template |
| `atlas-proxy/main.go` | HTTP server, verify-repair, best-of-K (2890 lines) |
| `atlas-proxy/agent.go` | Agent loop iteration (740 lines) |
| `atlas-proxy/tools.go` | 8 tools + tier classifier (905 lines) |
| `v3-service/main.py` | V3 pipeline orchestrator |
| `geometric-lens/main.py` | FastAPI server (26 endpoints) |
| `sandbox/executor_server.py` | 8-language executor |

## Documentation

| File | Content |
|------|---------|
| `docs/ARCHITECTURE.md` | Two-layer architecture, diagrams, data flow |
| `docs/API.md` | HTTP API reference (all 5 services) |
| `docs/CLI.md` | CLI usage and troubleshooting |
| `docs/CONFIGURATION.md` | All environment variables and settings |
| `docs/MAP.md` | Complete file-by-file repository map |
| `docs/SETUP.md` | Installation guide (Docker, bare-metal, K3s) |
| `docs/TROUBLESHOOTING.md` | Common issues and solutions |

## Tests

- `tests/infrastructure/` -- llama-server and sandbox connectivity
- `tests/integration/` -- End-to-end pipeline and training flows
- `tests/v3/` -- 22 unit test files for V3 pipeline modules
- Run: `make test` or `pytest tests/`

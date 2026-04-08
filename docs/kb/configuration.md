# Configuration and Environment Variables

## Port Assignments

| Service | Default Port | Protocol |
|---------|-------------|----------|
| llama-server | 8080 | HTTP (OpenAI-compatible) |
| atlas-proxy | 8090 | HTTP (OpenAI-compatible) |
| v3-service | 8070 | HTTP |
| geometric-lens | 8099 | HTTP (FastAPI) |
| sandbox | 30820 (host) / 8020 (container) | HTTP (FastAPI) |

## Docker Compose (.env)

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Purpose |
|----------|---------|--------|
| MODEL_PATH | ./models/Qwen3.5-9B-Q6_K.gguf | Path to GGUF model |
| CONTEXT_SIZE | 32768 | LLM context window |
| GPU_LAYERS | 99 | Layers to offload to GPU |
| PARALLEL_SLOTS | 4 | Concurrent LLM requests |

## K3s (atlas.conf)

Generated from `atlas.conf.example` by `scripts/generate-manifests.sh`.
Supports extended context (160K), KV cache quantization, flash attention,
mlock, NodePort exposure.

## VRAM Budget (RTX 5060 Ti 16GB)

| Config | Model VRAM | KV Cache | Total | Free |
|--------|-----------|----------|-------|------|
| 32K context | ~6.9 GB | ~1.3 GB | ~8.2 GB | ~7.8 GB |

All non-LLM services are CPU-only.

## Aider Integration

| File | Purpose |
|------|---------|
| `.aider.model.metadata.json` | Token limits (32K), cost ($0 local) |
| `.aider.model.settings.yml` | Whole-file edit format, streaming, temp 0.3 |

Aider connects to atlas-proxy on port 8090 as an OpenAI-compatible API.

## Python Package

`pyproject.toml` defines the `atlas` CLI entry point:
`atlas.cli.repl:run`. Requires Python >= 3.9.

Install: `pip install -e .`
Run: `atlas` (starts interactive REPL)

## Source References

- `docker-compose.yml` -- Service definitions and health checks
- `.env.example` -- Environment template
- `atlas.conf.example` -- K3s config template
- `docs/CONFIGURATION.md` -- Full configuration reference

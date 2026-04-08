# Deployment (Docker, Bare Metal, K3s)

## Docker Compose (Recommended)

Start all 5 services:

```
docker compose up -d
docker compose ps        # Wait for all "healthy"
atlas                    # Start coding
```

First run builds container images (several minutes). Subsequent starts
are fast.

### Startup Order

1. `llama-server` and `sandbox` start independently
2. `geometric-lens` and `v3-service` wait for llama-server healthy
3. `atlas-proxy` waits for all four services healthy

### Environment

Copy `.env.example` to `.env`. Defaults work if model is in `./models/`.
Key variables: MODEL_PATH, ports (8080/8099/8070/30820/8090),
CONTEXT_SIZE.

## Bare Metal

The `atlas` CLI (`pip install -e .`) talks directly to services on
default ports. The bash launcher can start all services as local
processes or detect a running Docker Compose stack.

## K3s

Manifests in `templates/` processed by `scripts/generate-manifests.sh`
from `atlas.conf`. Services deploy as pods in the `atlas` namespace
with NodePort exposure.

K3s entrypoint scripts (`inference/`) support:
- Extended context (160K)
- KV cache quantization (q8_0/q4_0)
- Flash attention
- mlock

Install: `scripts/install.sh` (K3s + GPU Operator)
Deploy: `scripts/deploy-9b.sh`
Verify: `scripts/verify-install.sh`

## VRAM Budget (RTX 5060 Ti 16GB, 32K context)

| Component | VRAM |
|-----------|------|
| Qwen3.5-9B-Q6_K weights | ~6.9 GB |
| KV cache (32K context) | ~1.3 GB |
| **Total llama-server** | **~8.2 GB** |
| geometric-lens | 0 (CPU) |
| v3-service | 0 (CPU) |
| sandbox | 0 (CPU) |
| atlas-proxy | 0 (Go, ~30 MB RAM) |
| **Free VRAM** | **~7.8 GB** |

## Prerequisites

- NVIDIA GPU (16GB+ VRAM) with proprietary drivers
- Docker with nvidia-container-toolkit (or Podman)
- Python 3.9+, pip, wget

# Troubleshooting Guide

## Quick Diagnostics

```
docker compose ps                                          # all services
curl -s http://localhost:8090/health | python3 -m json.tool # proxy (shows all)
nvidia-smi                                                 # GPU status
docker compose logs --tail 50                              # recent logs
```

Proxy health shows `inference`, `lens`, `sandbox` as true/false.

## Docker / GPU Issues

**GPU not detected in container:** Install nvidia-container-toolkit,
then `nvidia-ctk runtime configure --runtime=docker` and restart Docker.
Verify: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.

**First build fails (CUDA):** Needs ~5GB disk. llama-server builds inside
nvidia/cuda base image, so host CUDA not needed. Check disk space and network.

**SELinux blocking volumes (Fedora/RHEL):**
`chcon -Rt svirt_sandbox_file_t ~/models/`

**Port conflicts:** `lsof -i :8080` to find conflicts. All ports
configurable via `.env`.

## llama-server Issues

**Loading on CPU (~2 tok/s instead of ~50):** Ensure `--n-gpu-layers 99`.
Check nvidia-container-toolkit is configured.

**Model not found:** Verify `ls -la models/Qwen3.5-9B-Q6_K.gguf`.
Filename must match `ATLAS_MODEL_FILE` in `.env`.

**Out of VRAM / OOMKilled:** 9B Q6_K needs ~8.2 GB. Close other GPU
processes: `nvidia-smi` to check, kill if needed. Don't increase context
beyond 32K without checking VRAM.

**Grammar not enforced (outputs <think> tags):** Proxy sets
`response_format: json_object` when `ATLAS_AGENT_LOOP=1`. If using
llama-server directly, include it in your request.

**Context too small / truncation:** Should be 32768. Check `.env` or
bare-metal `--ctx-size` flag.

## Proxy Issues

**Agent loop not activating:** Set `ATLAS_AGENT_LOOP=1`. The `atlas`
launcher does this automatically.

**V3 not firing:** Requires all three: 50+ lines, 3+ logic indicators,
and v3-service reachable. Check `curl http://localhost:8070/health`.

**Truncation errors:** Model trying to write too much. Proxy rejects
write_file for existing files > 100 lines. Ask for targeted edits.

**Exploration budget warning:** 4+ consecutive reads without writing.
Be more specific about what to change.

## Geometric Lens Issues

**Unavailable / neutral scores (0.5):** Expected if model weights not
loaded. Service degrades gracefully. Download from HuggingFace or train.

**Embedding extraction fails:** Test: `curl http://localhost:8080/v1/embeddings`.
llama-server must be reachable from Lens.

## Sandbox Issues

**Unreachable:** Docker Compose: port 30820. Bare metal: port 8020.
Ensure same Docker network. Check `docker compose logs sandbox`.

**Execution timeout:** Default 30s, max 60s. Configurable via
`MAX_EXECUTION_TIME` env var.

**Language not supported:** 8 languages only: Python, JavaScript,
TypeScript, Go, Rust, C, C++, Bash.

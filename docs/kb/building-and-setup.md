# Building and Setup

Three deployment methods. Docker Compose is recommended and tested.

## Prerequisites (All Methods)

- NVIDIA GPU with 16GB+ VRAM, proprietary drivers (`nvidia-smi` works)
- Python 3.9+ with pip
- wget (for model download)
- ~20GB disk (7GB model + 5-8GB container images + working space)

## Docker Compose (Recommended)

```
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
mkdir -p models
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf \
     -O models/Qwen3.5-9B-Q6_K.gguf
pip install -e .
cp .env.example .env
docker compose up -d
docker compose ps          # wait for all "healthy"
atlas                      # start coding
```

**Additional prereqs:** Docker with nvidia-container-toolkit (or Podman).

### First Run Behavior

1. Builds 5 container images from source (~5-10 min for llama-server CUDA)
2. Loads 7GB model into GPU VRAM (~1-2 min)
3. Health checks all services
4. Subsequent starts are fast (seconds)

### Verify Installation

```
curl -s http://localhost:8080/health    # llama-server
curl -s http://localhost:8099/health    # geometric-lens
curl -s http://localhost:8070/health    # v3-service
curl -s http://localhost:30820/health   # sandbox
curl -s http://localhost:8090/health    # atlas-proxy (shows all statuses)
```

### Manage

```
docker compose logs -f <service>        # follow logs
docker compose down                     # stop (preserves images)
docker compose down --rmi all           # stop + remove images
git pull && docker compose build && docker compose up -d   # update
```

## Bare Metal

**Additional prereqs:** Go 1.24+, llama.cpp built with CUDA, Aider
(`pip install aider-chat`), Node.js 20+, Rust.

```
pip install -e .
cd atlas-proxy && go build -o ~/.local/bin/atlas-proxy-v2 . && cd ..
pip install -r geometric-lens/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pylint pytest pydantic
```

Start 5 services in separate terminals (see docs/SETUP.md for full
commands with env vars). Key difference: sandbox listens on **8020**
(not 30820) in bare-metal mode.

Alternatively, use the launcher: `cp atlas-launcher ~/.local/bin/atlas`
-- auto-detects running services, starts what's missing.

## K3s

**Additional prereqs:** K3s cluster, NVIDIA GPU Operator or device plugin,
Helm, Podman or Docker.

```
cp atlas.conf.example atlas.conf       # edit: model paths, GPU layers
sudo scripts/install.sh                 # automated: K3s + GPU + deploy
```

Or manually: `scripts/build-containers.sh`, `scripts/generate-manifests.sh`,
`kubectl apply -n atlas -f manifests/`, `scripts/verify-install.sh`.

## Geometric Lens Weights (Optional)

ATLAS works without Lens weights -- service returns neutral scores.
Pre-trained weights on [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).
Place in `geometric-lens/geometric_lens/models/`.

Training scripts: `scripts/retrain_cx.py`, `scripts/collect_lens_training_data.py`.

## Hardware Sizing

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB | 16 GB |
| System RAM | 14 GB | 16 GB+ |
| Disk | 15 GB | 25 GB |
| CPU | 4 cores | 8+ cores |

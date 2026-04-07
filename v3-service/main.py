"""V3 Pipeline Service -- stub.

Serves on port 8070. Provides /health so atlas-proxy can depend on it.
Full V3 pipeline implementation is tracked as a separate task.
"""

import os
from fastapi import FastAPI

app = FastAPI(title="V3 Pipeline Service")

INFERENCE_URL = os.getenv("ATLAS_INFERENCE_URL", "http://llama-server:8090")
LENS_URL = os.getenv("ATLAS_LENS_URL", "http://geometric-lens:8099")
SANDBOX_URL = os.getenv("ATLAS_SANDBOX_URL", "http://sandbox:8020")
MODEL_NAME = os.getenv("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "v3-pipeline",
        "model": MODEL_NAME,
        "stub": True,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)

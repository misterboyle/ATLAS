"""
Benchmark-specific configuration.

Reads settings from atlas.conf and provides defaults for benchmark operations.
"""

import os
import re
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the ATLAS submodule root directory (where benchmark/ lives).

    This always returns the atlas-src/ directory, regardless of where
    atlas.conf is located.  All benchmark-relative paths (datasets,
    results, etc.) are resolved from here.
    """
    return Path(__file__).resolve().parent.parent


def find_atlas_conf() -> 'Path | None':
    """Locate atlas.conf by walking up from the submodule root.

    Search order:
      1. ATLAS_CONF environment variable (absolute path override)
      2. Walk upward from the submodule root -- finds atlas.conf in
         atlas-src/ (legacy) or the parent repo root (preferred).

    Returns:
        Path to atlas.conf, or None if not found.
    """
    env_path = os.environ.get("ATLAS_CONF")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p

    current = get_project_root()
    while current != current.parent:
        candidate = current / "atlas.conf"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def parse_atlas_conf() -> dict:
    """
    Parse the atlas.conf file and return configuration as a dictionary.

    Returns:
        Dictionary of configuration values.
    """
    config = {}
    conf_path = find_atlas_conf()

    if conf_path is None:
        return config

    with open(conf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value

    return config


class BenchmarkConfig:
    """Configuration for benchmark operations."""

    def __init__(self):
        """Initialize configuration from atlas.conf and environment."""
        self._conf = parse_atlas_conf()
        self._root = get_project_root()

    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return self._root

    @property
    def benchmark_dir(self) -> Path:
        """Benchmark module directory."""
        return self._root / "benchmark"

    @property
    def datasets_dir(self) -> Path:
        """Datasets directory."""
        return self.benchmark_dir / "datasets"

    @property
    def cache_dir(self) -> Path:
        """Dataset cache directory.

        Uses a shared user-level cache so all worktrees and the main
        repo find the same downloaded datasets.  Override with the
        ATLAS_DATASET_CACHE environment variable.

        Default: $XDG_CACHE_HOME/atlas/datasets  (~/.cache/atlas/datasets)
        """
        env = os.environ.get('ATLAS_DATASET_CACHE')
        if env:
            return Path(env)
        xdg = os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))
        return Path(xdg) / 'atlas' / 'datasets' 

    @property
    def custom_dir(self) -> Path:
        """Custom tasks directory."""
        return self.benchmark_dir / "custom"

    @property
    def results_dir(self) -> Path:
        """Results output directory."""
        return self.benchmark_dir / "results"

    @property
    def submissions_dir(self) -> Path:
        """Submissions directory."""
        return self.results_dir / "submissions"

    @property
    def llama_url(self) -> str:
        """URL for llama-server."""
        # Check environment first
        url = os.environ.get("LLAMA_URL")
        if url:
            return url

        # Check if running in cluster
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return "http://llama-service:8000"

        # Use NodePort from config
        port = self._conf.get("ATLAS_LLAMA_NODEPORT", "32735")
        return f"http://localhost:{port}"

    @property
    def llama_api_url(self) -> str:
        """URL for llama-server OpenAI-compatible API."""
        return f"{self.llama_url}/v1"

    @property
    def model_name(self) -> str:
        """Main model filename."""
        return self._conf.get("ATLAS_MAIN_MODEL", "Qwen3.5-9B-Opus-Distilled-v2-Q8_0.gguf")

    @property
    def default_timeout_seconds(self) -> int:
        """Default timeout for code execution."""
        return int(self._conf.get("ATLAS_SANDBOX_TIMEOUT", "30"))

    @property
    def default_memory_limit_mb(self) -> int:
        """Default memory limit for code execution."""
        return 512

    @property
    def default_k(self) -> int:
        """Default number of attempts per task."""
        return 1

    @property
    def default_temperature_pass1(self) -> float:
        """Temperature for pass@1 (greedy decoding)."""
        return 0.0

    @property
    def default_temperature_passk(self) -> float:
        """Temperature for pass@k evaluation."""
        return 0.8

    @property
    def gpu_tdp_watts(self) -> float:
        """GPU TDP in watts (RTX 5060 Ti)."""
        return 180.0

    @property
    def gpu_cost_usd(self) -> float:
        """Estimated GPU cost in USD."""
        return 450.0

    @property
    def gpu_lifetime_hours(self) -> float:
        """Expected GPU lifetime in hours (5 years, 8 hours/day)."""
        return 5 * 365 * 8

    # Cloud API pricing (per 1M tokens as of 2024)
    @property
    def cloud_pricing(self) -> dict:
        """Cloud API pricing per 1M tokens (input/output)."""
        return {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet": {"input": 3.0, "output": 15.0},
            "claude-haiku": {"input": 0.25, "output": 1.25}
        }

    # Published baselines for comparison
    @property
    def qwen3_14b_baselines(self) -> dict:
        """Published Qwen3-14B baseline scores."""
        return {
            "humaneval_pass1": 0.67,       # ~65-70%
            "mbpp_pass1": 0.734,           # 73.4% per tech report (3-shot)
            "humaneval_plus_pass1": 0.61,  # EvalPlus leaderboard estimate
            "mbpp_plus_pass1": 0.65,       # EvalPlus leaderboard estimate
            "livecodebench_pass1": 0.20,   # Placeholder — will update after first run
            "scicode_pass1": 0.10,         # Placeholder — will update after first run
        }

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = BenchmarkConfig()

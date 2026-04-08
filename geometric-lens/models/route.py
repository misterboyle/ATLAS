"""Routing models for difficulty-based LLM pipeline selection."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

# Context and token budgets
ROUTE_CONTEXT_BUDGET = 8000
ROUTE_MAX_TOKENS = 4096
ROUTE_RETRY_BUDGET = 3


class DifficultyBin(str, Enum):
    """Task difficulty classification."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Route(str, Enum):
    """Available routing targets."""
    DIRECT = "direct"
    RAG = "rag"
    RAG_VERIFIED = "rag_verified"
    V3_FAST = "v3_fast"
    V3_THOROUGH = "v3_thorough"


# Fallback chain: if primary route fails, try next
FALLBACK_CHAIN = [
    Route.V3_THOROUGH,
    Route.V3_FAST,
    Route.RAG_VERIFIED,
    Route.RAG,
    Route.DIRECT,
]


def difficulty_to_bin(score: float) -> DifficultyBin:
    """Convert a 0.0--1.0 difficulty score to a DifficultyBin."""
    if score < 0.2:
        return DifficultyBin.TRIVIAL
    elif score < 0.4:
        return DifficultyBin.EASY
    elif score < 0.6:
        return DifficultyBin.MEDIUM
    elif score < 0.8:
        return DifficultyBin.HARD
    return DifficultyBin.EXPERT


@dataclass
class SignalBundle:
    """Collected signals for routing decision."""
    prompt_length: int = 0
    estimated_difficulty: float = 0.0
    difficulty_bin: DifficultyBin = DifficultyBin.MEDIUM
    has_code: bool = False
    has_tests: bool = False
    language: str = ""
    task_type: str = ""
    context_files: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

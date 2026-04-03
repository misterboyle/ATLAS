"""Route selection models for the Confidence Router."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class Route(str, Enum):
    CACHE_HIT = "cache_hit"
    FAST_PATH = "fast_path"
    STANDARD = "standard"
    HARD_PATH = "hard_path"


class DifficultyBin(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


ROUTE_COSTS: Dict[Route, float] = {
    Route.CACHE_HIT: 0.1,
    Route.FAST_PATH: 0.5,
    Route.STANDARD: 1.0,
    Route.HARD_PATH: 3.0,
}

ROUTE_RETRY_BUDGET: Dict[Route, int] = {
    Route.CACHE_HIT: 1,
    Route.FAST_PATH: 2,
    Route.STANDARD: 3,
    Route.HARD_PATH: 5,
}

FALLBACK_CHAIN: Dict[Route, Optional[Route]] = {
    Route.CACHE_HIT: Route.FAST_PATH,
    Route.FAST_PATH: Route.STANDARD,
    Route.STANDARD: Route.HARD_PATH,
    Route.HARD_PATH: None,
}

ROUTE_CONTEXT_BUDGET: Dict[Route, int] = {
    Route.CACHE_HIT: 5000,
    Route.FAST_PATH: 2000,
    Route.STANDARD: 2000,
    Route.HARD_PATH: 3000,
}

ROUTE_MAX_TOKENS: Dict[Route, int] = {
    Route.CACHE_HIT: 1000,
    Route.FAST_PATH: 4000,
    Route.STANDARD: 6000,
    Route.HARD_PATH: 8000,
}


def difficulty_to_bin(difficulty: float) -> DifficultyBin:
    """Map a continuous difficulty score to a discrete bin."""
    if difficulty < 0.3:
        return DifficultyBin.EASY
    elif difficulty < 0.6:
        return DifficultyBin.MEDIUM
    else:
        return DifficultyBin.HARD


class SignalBundle(BaseModel):
    """Routing signals from cache, retrieval, and query analysis."""
    pattern_cache_score: float = 0.0
    retrieval_confidence: float = 0.0
    query_complexity: float = 0.0
    geometric_energy: float = 0.0


class RouteDecision(BaseModel):
    """Result of route selection."""
    route: Route
    difficulty_score: float
    difficulty_bin: DifficultyBin
    retry_budget: int
    context_budget: int = 2000
    max_tokens: int = 6000
    signals: SignalBundle
    thompson_samples: Optional[Dict[str, float]] = None
    cache_hit_available: bool = False

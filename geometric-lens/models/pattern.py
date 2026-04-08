"""Pattern models for code pattern caching and matching."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

# Half-lives for pattern decay (hours)
HALF_LIVES: Dict[str, float] = {
    "exact": 168.0,       # 1 week
    "structural": 336.0,  # 2 weeks
    "semantic": 720.0,    # 1 month
    "heuristic": 48.0,    # 2 days
}


class PatternType(str, Enum):
    """Type of code pattern."""
    EXACT = "exact"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    HEURISTIC = "heuristic"


class PatternTier(str, Enum):
    """Quality tier of a pattern."""
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    STALE = "stale"


@dataclass
class PatternScore:
    """Score for a pattern match."""
    similarity: float = 0.0
    recency: float = 0.0
    success_rate: float = 0.0
    composite: float = 0.0

    def compute_composite(
        self, weights: Optional[Dict[str, float]] = None
    ) -> float:
        w = weights or {
            "similarity": 0.5,
            "recency": 0.2,
            "success_rate": 0.3,
        }
        self.composite = (
            w.get("similarity", 0.5) * self.similarity
            + w.get("recency", 0.2) * self.recency
            + w.get("success_rate", 0.3) * self.success_rate
        )
        return self.composite


@dataclass
class Pattern:
    """A cached code pattern."""
    id: str = ""
    pattern_type: PatternType = PatternType.HEURISTIC
    tier: PatternTier = PatternTier.BRONZE
    task_signature: str = ""
    solution_code: str = ""
    language: str = "python"
    tags: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

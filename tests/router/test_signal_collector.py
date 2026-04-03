"""Tests for signal_collector -- routing signal computation."""
from types import SimpleNamespace
import pytest
from router.signal_collector import (
    compute_query_complexity,
    compute_pattern_cache_score,
    compute_retrieval_confidence,
    collect_signals,
)
from models.route import SignalBundle


def _pattern(similarity, tier="stm"):
    """Helper to build a scored-pattern-like object."""
    return SimpleNamespace(
        similarity=similarity,
        pattern=SimpleNamespace(tier=SimpleNamespace(value=tier)),
    )


class TestComputeQueryComplexity:
    def test_empty_string(self):
        assert compute_query_complexity("") == pytest.approx(0.0, abs=0.01)

    def test_short_query(self):
        v = compute_query_complexity("hello world")
        assert 0.0 <= v <= 1.0
        assert v < 0.1

    def test_long_query(self):
        q = " ".join(["word"] * 500)
        v = compute_query_complexity(q)
        assert v > 0.3

    def test_code_blocks_increase(self):
        q = "```python\ncode\n```\n" * 3
        v = compute_query_complexity(q)
        assert v > compute_query_complexity("simple question")

    def test_clamped_at_one(self):
        q = " ".join(["word"] * 2000) + "\n" * 200 + "```" * 20
        assert compute_query_complexity(q) == 1.0

    def test_multiline(self):
        q = "line1\nline2\nline3"
        v = compute_query_complexity(q)
        assert v > compute_query_complexity("line1")


class TestComputePatternCacheScore:
    def test_empty_returns_zero(self):
        assert compute_pattern_cache_score(None) == 0.0
        assert compute_pattern_cache_score([]) == 0.0

    def test_single_pattern(self):
        s = compute_pattern_cache_score([_pattern(0.8)])
        assert s == pytest.approx(0.8, abs=0.01)

    def test_persistent_tier_discounted(self):
        s = compute_pattern_cache_score([_pattern(0.8, "persistent")])
        assert s == pytest.approx(0.24, abs=0.01)  # 0.8 * 0.3

    def test_best_of_top3(self):
        pats = [_pattern(0.3), _pattern(0.9), _pattern(0.5), _pattern(0.99)]
        s = compute_pattern_cache_score(pats)
        # Only top 3 considered: 0.3, 0.9, 0.5 -> best=0.9
        assert s == pytest.approx(0.9, abs=0.01)

    def test_clamped_at_one(self):
        assert compute_pattern_cache_score([_pattern(1.5)]) == 1.0


class TestComputeRetrievalConfidence:
    def test_empty_returns_zero(self):
        assert compute_retrieval_confidence([]) == 0.0

    def test_chunks_with_scores(self):
        chunks = [{"score": 5.0}, {"score": 2.0}]
        v = compute_retrieval_confidence(chunks)
        # count_signal = min(1, 2/5) = 0.4
        # score_signal = min(1, 5.0/5.0) = 1.0
        # result = 0.4*0.4 + 0.6*1.0 = 0.76
        assert v == pytest.approx(0.76, abs=0.01)

    def test_chunks_without_scores(self):
        chunks = [{}, {}, {}, {}, {}]
        v = compute_retrieval_confidence(chunks)
        # count_signal = 5/5 = 1.0, no scores -> score_signal = count_signal = 1.0
        # result = 0.4*1.0 + 0.6*1.0 = 1.0
        assert v == pytest.approx(1.0, abs=0.01)

    def test_single_chunk_low_score(self):
        chunks = [{"score": 1.0}]
        v = compute_retrieval_confidence(chunks)
        # count_signal = 1/5 = 0.2
        # score_signal = min(1, 1.0/5.0) = 0.2
        # result = 0.4*0.2 + 0.6*0.2 = 0.2
        assert v == pytest.approx(0.2, abs=0.01)


class TestCollectSignals:
    def test_returns_signal_bundle(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "router.signal_collector.compute_geometric_energy",
                lambda q: 0.0,
            )
            b = collect_signals("test query")
            assert isinstance(b, SignalBundle)
            assert b.geometric_energy == 0.0
            assert b.query_complexity >= 0.0

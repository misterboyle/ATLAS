"""Tests for route_selector -- Thompson Sampling route selection."""
from unittest.mock import MagicMock, patch
import pytest
from models.route import (
    Route, DifficultyBin, SignalBundle, RouteDecision,
    ROUTE_COSTS, ROUTE_RETRY_BUDGET,
)
from router.route_selector import (
    select_route, _load_thompson_state, _key,
    get_all_thompson_states, reset_thompson_state,
)


class TestKey:
    def test_format(self):
        k = _key(DifficultyBin.EASY, Route.STANDARD, "alpha")
        assert k == "confidence_router:thompson:easy:standard:alpha"


class TestLoadThompsonState:
    def test_empty_redis_uniform_priors(self, mock_redis):
        st = _load_thompson_state(mock_redis, DifficultyBin.EASY)
        for route in Route:
            assert st[route] == (1.0, 1.0)

    def test_existing_counts_add_prior(self, mock_redis):
        mock_redis._store[
            _key(DifficultyBin.EASY, Route.STANDARD, "alpha")
        ] = "5.0"
        mock_redis._store[
            _key(DifficultyBin.EASY, Route.STANDARD, "beta")
        ] = "3.0"
        st = _load_thompson_state(mock_redis, DifficultyBin.EASY)
        assert st[Route.STANDARD] == (6.0, 4.0)


class TestSelectRoute:
    def test_returns_route_decision(self, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.5)
        assert isinstance(r, RouteDecision)
        assert r.difficulty_bin == DifficultyBin.MEDIUM

    @patch("random.betavariate", return_value=0.8)
    def test_cache_hit_when_available_low_diff(self, _m, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.1, cache_hit_available=True)
        assert r.route == Route.CACHE_HIT

    @patch("random.betavariate", return_value=0.8)
    def test_cache_hit_excluded_unavailable(self, _m, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.1, cache_hit_available=False)
        assert r.route != Route.CACHE_HIT

    @patch("random.betavariate", return_value=0.8)
    def test_cache_hit_excluded_high_diff(self, _m, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.5, cache_hit_available=True)
        assert r.route != Route.CACHE_HIT

    @patch("random.betavariate", return_value=0.8)
    def test_fast_path_penalized_hard(self, _m, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.8)
        # FAST 0.8/0.5*0.3=0.48 < STD 0.8/1.0=0.8
        assert r.route == Route.STANDARD

    @patch("random.betavariate", return_value=0.8)
    def test_hard_path_penalized_easy(self, _m, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.1, cache_hit_available=False)
        # FAST 0.8/0.5=1.6 > STD 0.8/1.0=0.8 > HARD 0.8/3.0*0.3=0.08
        assert r.route == Route.FAST_PATH

    def test_redis_failure_defaults_standard(self, default_signals):
        bad = MagicMock()
        bad.get = MagicMock(side_effect=ConnectionError("down"))
        r = select_route(bad, default_signals, 0.5)
        assert r.route == Route.STANDARD

    def test_retry_budget_matches(self, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.5)
        assert r.retry_budget == ROUTE_RETRY_BUDGET[r.route]

    def test_thompson_samples_populated(self, mock_redis, default_signals):
        r = select_route(mock_redis, default_signals, 0.5)
        assert r.thompson_samples is not None
        assert len(r.thompson_samples) > 0

    @patch("random.betavariate", return_value=0.8)
    def test_cost_weighting_prefers_cheap(self, _m, mock_redis, default_signals):
        # With equal success prob, cheapest feasible route wins
        r = select_route(mock_redis, default_signals, 0.4)
        eff = {k: 0.8 / ROUTE_COSTS[Route(k)] for k in r.thompson_samples}
        best_key = max(eff, key=eff.get)
        assert r.route.value == best_key


class TestGetAllThompsonStates:
    def test_returns_all_bins_routes(self, mock_redis):
        st = get_all_thompson_states(mock_redis)
        for db in DifficultyBin:
            assert db.value in st
            for rt in Route:
                assert rt.value in st[db.value]
                assert "alpha" in st[db.value][rt.value]
                assert "mean_success_rate" in st[db.value][rt.value]


class TestResetThompsonState:
    def test_reset_clears_state(self, mock_redis):
        k = _key(DifficultyBin.EASY, Route.STANDARD, "alpha")
        mock_redis._store[k] = "10.0"
        reset_thompson_state(mock_redis)
        st = _load_thompson_state(mock_redis, DifficultyBin.EASY)
        assert st[Route.STANDARD] == (1.0, 1.0)

"""Tests for fallback_chain -- escalation logic."""
import pytest
from models.route import Route, ROUTE_RETRY_BUDGET
from router.fallback_chain import get_fallback_route, get_escalation_path


class TestGetFallbackRoute:
    def test_cache_hit_to_fast(self):
        assert get_fallback_route(Route.CACHE_HIT) == Route.FAST_PATH

    def test_fast_to_standard(self):
        assert get_fallback_route(Route.FAST_PATH) == Route.STANDARD

    def test_standard_to_hard(self):
        assert get_fallback_route(Route.STANDARD) == Route.HARD_PATH

    def test_hard_path_terminal(self):
        assert get_fallback_route(Route.HARD_PATH) is None

    def test_full_chain(self):
        cur = Route.CACHE_HIT
        chain = [cur]
        while True:
            nxt = get_fallback_route(cur)
            if nxt is None:
                break
            chain.append(nxt)
            cur = nxt
        assert chain == [
            Route.CACHE_HIT, Route.FAST_PATH,
            Route.STANDARD, Route.HARD_PATH,
        ]


class TestGetEscalationPath:
    def test_from_cache_hit(self):
        path = get_escalation_path(Route.CACHE_HIT)
        assert len(path) == 4
        routes = [r for r, _ in path]
        assert routes == [
            Route.CACHE_HIT, Route.FAST_PATH,
            Route.STANDARD, Route.HARD_PATH,
        ]

    def test_from_standard(self):
        path = get_escalation_path(Route.STANDARD)
        assert len(path) == 2
        assert path[0] == (Route.STANDARD, ROUTE_RETRY_BUDGET[Route.STANDARD])
        assert path[1] == (Route.HARD_PATH, ROUTE_RETRY_BUDGET[Route.HARD_PATH])

    def test_from_hard_path(self):
        path = get_escalation_path(Route.HARD_PATH)
        assert len(path) == 1
        assert path[0] == (Route.HARD_PATH, ROUTE_RETRY_BUDGET[Route.HARD_PATH])

    def test_budgets_increase(self):
        path = get_escalation_path(Route.CACHE_HIT)
        budgets = [b for _, b in path]
        assert budgets == sorted(budgets)

"""Shared fixtures for router tests."""
import pytest
from models.route import SignalBundle


class DictRedis:
    """Minimal Redis mock backed by a dict."""
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def delete(self, *keys):
        for k in keys:
            self._store.pop(k, None)

    def incr(self, key, amount=1):
        self._store[key] = int(self._store.get(key) or 0) + amount
        return self._store[key]

    def incrbyfloat(self, key, amount):
        self._store[key] = str(float(self._store.get(key) or 0) + amount)
        return self._store[key]

    def pipeline(self):
        return DictPipeline(self._store)


class DictPipeline:
    """Minimal pipeline mock."""
    def __init__(self, store):
        self._s = store
        self._ops = []

    def set(self, key, value):
        self._ops.append(('set', key, value))

    def delete(self, key):
        self._ops.append(('del', key))

    def incr(self, key, amount=1):
        self._ops.append(('incr', key, amount))

    def incrbyfloat(self, key, amount):
        self._ops.append(('incrf', key, amount))

    def execute(self):
        for op in self._ops:
            if op[0] == 'set':
                self._s[op[1]] = op[2]
            elif op[0] == 'del':
                self._s.pop(op[1], None)
            elif op[0] == 'incr':
                self._s[op[1]] = int(self._s.get(op[1]) or 0) + op[2]
            elif op[0] == 'incrf':
                self._s[op[1]] = str(
                    float(self._s.get(op[1]) or 0) + op[2]
                )
        self._ops.clear()


@pytest.fixture
def mock_redis():
    return DictRedis()


@pytest.fixture
def default_signals():
    return SignalBundle(
        pattern_cache_score=0.5, retrieval_confidence=0.5,
        query_complexity=0.5, geometric_energy=0.0,
    )

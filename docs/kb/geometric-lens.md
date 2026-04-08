# Geometric Lens (Scoring, RAG, Routing)

Neural scoring system that evaluates code quality without executing it.
Runs entirely on CPU. Also serves as the RAG API for project indexing,
retrieval, confidence routing, and pattern caching.

## Scoring Models

### C(x) Cost Field

- Architecture: 4096 -> 512 -> 128 -> 1 MLP (SiLU + Softplus)
- Training: 597 LCB embeddings (504 PASS, 93 FAIL), contrastive ranking
- Performance: Val AUC 0.9467, separation 2.04x
- Normalization: `1 / (1 + exp(-(energy - 19.0) / 2.0))` -> [0, 1]
- Parameters: 2,163,457 (~8.7 MB)
- Source: `geometric-lens/geometric_lens/cost_field.py`

### G(x) XGBoost

- Architecture: PCA(4096 -> 128) + XGBoost classifier
- Training: 13,398 embeddings (4,835 PASS, 8,563 FAIL)
- PCA: 80.8% variance retained
- Source: `geometric-lens/geometric_lens/metric_tensor.py`

### Combined Verdict

| Score | Verdict |
|-------|---------|
| >= 0.7 | likely_correct |
| >= 0.3 | uncertain |
| < 0.3 | likely_incorrect |

### Graceful Degradation

Model weights (.pt, .pkl) are not committed -- built during training
and baked into container images. When absent, C(x) returns neutral
energy, G(x) returns `gx_score: 0.5, verdict: "unavailable"`.

## Training Infrastructure

- **EWC:** Fisher information prevents catastrophic forgetting
- **Replay Buffer:** Domain-stratified, 30% old / 70% new
- Source: `geometric-lens/geometric_lens/ewc.py`,
  `geometric-lens/geometric_lens/replay_buffer.py`

## RAG / PageIndex V2

- **Indexing:** tree-sitter AST parsing -> hierarchical tree ->
  BM25 inverted index + LLM summaries -> JSON persistence
- **Retrieval:** BM25 search (min_score=0.1, top_k=20) +
  LLM-guided tree traversal (max_depth=6, max_calls=40)
- **Hybrid:** Routes bm25_first / tree_only / both
- Source: `geometric-lens/indexer/`, `geometric-lens/retriever/`

## Confidence Router

4 routes with cost-weighted Thompson Sampling:

| Route | Cost | K |
|-------|------|---|
| CACHE_HIT | 1 | 0 |
| FAST_PATH | 50 | 1 |
| STANDARD | 300 | 5 |
| HARD_PATH | 1500 | 20 |

Signals: pattern_cache, retrieval_confidence, query_complexity,
geometric_energy. Fallback chain escalates on failure.
Source: `geometric-lens/router/`

## Pattern Cache

Redis-backed with three tiers: STM (100 entries), LTM, PERSISTENT.
BM25 matching over summaries. Ebbinghaus decay scoring.
Co-occurrence graph for linked pattern retrieval.
Source: `geometric-lens/cache/`

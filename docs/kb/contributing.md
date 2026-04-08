# Contributing and Development Workflow

## Development Setup

```
git clone https://github.com/<your-username>/atlas.git
cd atlas
git checkout -b feature/your-feature-name
cp .env.example .env
python tests/validate_tests.py          # verify tests pass first
```

## Code Style

### Python
- PEP 8, type hints on function signatures
- Docstrings on public functions
- Max line length: 100 characters

### Bash
- Use `set -euo pipefail`
- Quote variables: `"$var"`
- Use `[[` for conditionals
- Run shellcheck

### YAML/Kubernetes
- 2-space indentation
- Resource limits on all containers

### Documentation
- Markdown for everything
- Include code examples
- Update ToC when adding sections

## Commit Message Format

```
component: short description (50 chars max)

Longer description if needed. Explain what and why.

Fixes #123
```

Examples:
- `geometric-lens: add project caching for faster queries`
- `sandbox: increase default timeout to 90s`
- `docs: add GPU troubleshooting section`

## Running Tests

```
python tests/validate_tests.py           # all tests
pytest tests/v3/test_plan_search.py -v   # specific test
pytest tests/ --cov=. --cov-report=html  # with coverage
```

Test structure:
- `tests/infrastructure/` -- llama-server and sandbox connectivity
- `tests/integration/` -- end-to-end pipeline and training
- `tests/v3/` -- 22 unit test files for V3 modules

New features must include tests. Bug fixes need regression tests.

## Pull Request Workflow

1. Fork and create feature branch
2. Make focused, atomic commits
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Open PR with clear description and linked issues
6. Address review feedback, request re-review

## Architecture Decisions

For architectural changes:
1. Document the problem and proposed solution
2. List alternatives considered
3. Explain trade-offs
4. Consider backwards compatibility
5. Update architecture documentation

## License

ATLAS Source Available License v1.0. Contributions accepted under
the same license. You retain copyright but grant the maintainer a
perpetual royalty-free license to use, modify, and distribute.

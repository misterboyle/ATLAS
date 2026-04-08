# Sandbox (Isolated Code Execution)

Isolated code execution service supporting 8 languages.

## Language Executors

| Language | Aliases | Toolchain |
|----------|---------|----------|
| Python | py, python3 | pylint (0-10) + pytest |
| JavaScript | js, node | Node.js 20 |
| TypeScript | ts | tsc --noEmit + tsx |
| Go | golang | go build + run (Go 1.22) |
| Rust | rs | rustc + run (stable) |
| C | -- | gcc -Wall |
| C++ | c++ | g++ -Wall |
| Bash | sh, shell | bash -n + run |

## Build Verification (atlas-proxy side)

Per-language syntax checks run before sandbox testing:
- Python: `py_compile`
- TypeScript: `tsc --noEmit`
- JavaScript: `node --check`
- Go: `go build`
- Rust: `cargo check`
- C/C++: `gcc/g++ -fsyntax-only`
- Shell: `bash -n`

Framework overrides apply for Next.js, React, Flask, Django, Express.

## Error Classification

15 error types: SyntaxError, NameError, TypeError, CompileError,
Timeout, RuntimeError, ImportError, AssertionError, etc.

## Resource Limits

| Limit | Value |
|-------|-------|
| Max execution time | 60 seconds |
| Max memory | 512 MB |
| Workspace | /tmp/sandbox (tmpfs) |
| stdout truncation | 4,000 chars |
| stderr truncation | 2,000 chars |

## Source

- `sandbox/executor_server.py` -- FastAPI server, all executors
- `sandbox/Dockerfile` -- Container (Python, Node, Go, Rust, gcc)

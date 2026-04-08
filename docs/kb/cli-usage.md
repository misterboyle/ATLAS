# CLI Usage and Workflow

## Launching

```
cd /path/to/your/project
atlas                              # interactive session
atlas somefile.py                  # add file to chat on launch
atlas --message "fix the bug"      # non-interactive (runs and exits)
echo "solve this" | atlas          # pipe mode
```

Auto-detects Docker Compose stack or starts bare-metal services.
All args after `atlas` pass through to Aider.

## Health Check Timeouts

| Service | Timeout |
|---------|---------|
| llama-server | 120s (model loading) |
| Geometric Lens | 30s |
| V3 Pipeline | 15s |
| Proxy v2 | 30s |

Already-running services are skipped. Logs go to `logs/`.

## Streaming Status Icons

| Icon | Tool |
|------|------|
| (pencil) | write_file |
| (pen) | edit_file |
| (wrench) | run_command |
| (book) | read_file |
| (search) | search_files |
| (folder) | list_directory |
| (clipboard) | plan_tasks |

Result indicators: checkmark = success, X = failure.

## V3 Pipeline in Output

T2+ files show V3 stages inline:
- [probe] Generating probe candidate...
- [probe_scored] C(x)=0.72
- [plansearch] Generating 3 plans...
- [sandbox_test] Testing candidates...
- V3 complete: phase and candidate count

## Aider Commands

| Command | Description |
|---------|-------------|
| /add <file> | Add file to chat context |
| /drop <file> | Remove file from context |
| /clear | Clear chat history |
| /tokens | Show token usage |
| /undo | Undo last change |
| /run <cmd> | Run shell command |
| /help | Show all commands |

## Python REPL (Fallback)

If no Docker stack or launcher found, falls back to Python REPL:
- /solve <file> -- solve coding problem
- /bench -- run benchmarks
- /status -- check service health
- /quit -- exit

Generation params: max_tokens=8192, temperature=0.6, top_k=20, top_p=0.95.

## Tips for Best Results

1. **Be specific** -- detailed prompts work better than vague ones
2. **Provide file context** -- /add files so ATLAS can read them
3. **Complex tasks take longer** -- V3 adds 2-5 min but better code
4. **Watch the terminal** -- streaming shows every step in real-time
5. **Ask for targeted edits** -- proxy rejects write_file on files > 100
   lines; ask for specific changes instead of full rewrites

## What ATLAS Does Well

- Single-file creation (Python, Rust, Go, C, shell)
- Multi-file project scaffolding (Next.js, Flask, Express)
- Bug fixes via surgical edit_file
- Code analysis and explanation

## Current Limitations

- Large codebases (50+ files) -- 32K context limit
- Visual/CSS verification -- sandbox cannot check rendering
- Interactive UIs -- no browser testing
- Adding features to existing projects -- ~67% reliability on 9B model

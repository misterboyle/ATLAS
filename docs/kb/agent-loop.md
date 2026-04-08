# Agent Loop (atlas-proxy)

## Grammar-Constrained Output

Every LLM response is forced into exactly one of three JSON shapes:

```
{"type": "tool_call", "name": "<tool>", "args": {...}}
{"type": "text", "content": "<message>"}
{"type": "done", "summary": "<summary>"}
```

The JSON schema uses `oneOf` with `additionalProperties: false`. Token
generation is grammar-constrained at the llama-server level -- the model
cannot produce invalid JSON.

## 8 Tools

| Tool | Purpose | Read-only |
|------|---------|-----------|
| read_file | Read file contents (offset/limit) | Yes |
| write_file | Create/overwrite (routes to V3 for T2) | No |
| edit_file | Replace exact string (old_str/new_str) | No |
| delete_file | Delete file or empty directory | No |
| run_command | Shell command (5 min timeout) | No |
| search_files | Regex search (max 200 matches) | Yes |
| list_directory | List contents with type and size | Yes |
| plan_tasks | Parallel tasks with dependency graph | No |

## Per-File Tier Classification

| Tier | Max Turns | Behavior |
|------|-----------|----------|
| T0 (Conversational) | 5 | Text response only |
| T1 (Simple) | 30 | Direct write, no V3 |
| T2 (Feature) | 30 | V3 pipeline fires |
| T3 (Hard) | 60 | V3 pipeline fires |

**Always T1:** Config files (.json, .yaml, .toml, .env, etc.), docs (.md),
shell scripts (.sh), CSS, short files (< 50 lines).

**T2:** Files with 50+ lines AND 3+ logic indicators (function defs,
control flow, API patterns, state management, JSX).

## Safety Limits

| Limit | Value |
|-------|-------|
| Conversation trim | 12 messages max |
| write_file for existing | Reject if > 100 lines |
| Error loop breaker | 3 consecutive failures |
| Exploration budget warn | 4 consecutive reads |
| Exploration budget skip | 5+ consecutive reads |
| Command stdout cap | 8,000 chars |
| Command stderr cap | 4,000 chars |
| Search results cap | 200 matches |
| File search skip | Files > 1 MB |

## Source Files

- `atlas-proxy/agent.go` -- Agent loop, LLM dispatch, exploration budget
- `atlas-proxy/tools.go` -- Tool definitions, tier classifier
- `atlas-proxy/grammar.go` -- JSON schema + GBNF grammar
- `atlas-proxy/types.go` -- AgentContext, tiers, types

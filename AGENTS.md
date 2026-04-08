# AGENTS.md -- Remote Agent Guidelines

## 1. Read Architecture Before Working

Before making any changes, query the knowledge base:

```
NEED_KB: architecture overview
NEED_KB: <component you are about to change>
```

The KB is searchable and instant -- always cheaper than guessing.
Use `NEED_KB_LIST:` to see all available sections.

## 2. Session Completion -- Refspec Push (Critical)

You work in an isolated worktree. **Never push your agent branch as a
remote branch.** Use the refspec pattern to push directly to the parent:

```bash
PARENT=<parent-branch>   # from workspace_info.parent_branch
git add -A && git commit -m "<description>"
git fetch origin
git rebase origin/"$PARENT"
make test
git push origin HEAD:"$PARENT"
```

If push is rejected (another agent pushed first), retry:

```bash
git fetch origin && git rebase origin/"$PARENT" && make test && git push origin HEAD:"$PARENT"
```

Rules:
- **You** must push -- never say "ready when you are"
- Commit, rebase, test, and push in the **same** iteration
- Do NOT checkout or merge in the main repo root
- Do NOT push `agent/*` as remote branches
- Do NOT rewrite history on shared branches (no force-push, no interactive rebase)

## 3. Document Issues with Beads

Use `bd` for all issue tracking. File issues for anything you discover:

```bash
bd create "<description>" --type bug --priority 3
```

Before starting work, always claim:

```bash
bd update <id> --claim
```

If the claim fails (another agent took it), pick a different bead.

## 4. Keep Architecture Docs Current

If you change the architecture (add/remove services, alter data flow,
change module boundaries), update the relevant KB section under `docs/kb/`
and its entry in `docs/kb/_index.json`. Set `coupled_files` so future
changes trigger a review flag. Always `git add -A` to capture
`docs/kb/_usage.json` updates.

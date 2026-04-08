# AGENTS.md -- Supplemental Agent Guidelines

These supplement the bootstrap rules you already received.
Do NOT duplicate effort -- bootstrap covers bd usage, KB search
directives, claim-before-work, and proactive issue filing.

## 1. Read Architecture First

Before changing any code, query `NEED_KB: architecture overview` and
`NEED_KB: <component you are changing>`. Understand the two-layer
architecture (agent loop + V3 pipeline) and the 5-service stack before
making design decisions.

## 2. Refspec Push (Worktree-Specific)

Bootstrap says "merge into parent branch and push" -- here is the
exact pattern for worktree agents:

```bash
PARENT=<parent-branch>   # from workspace_info.parent_branch
git add -A && git commit -m "<description>"
git fetch origin
git rebase origin/"$PARENT"
make test
git push origin HEAD:"$PARENT"
```

Retry on rejection (another agent pushed first):

```bash
git fetch origin && git rebase origin/"$PARENT" && make test && git push origin HEAD:"$PARENT"
```

Critical rules not in bootstrap:
- Do NOT push `agent/*` as remote branches -- use `HEAD:$PARENT` refspec
- Do NOT checkout or merge in the main repo root
- Do NOT rewrite history on shared branches (no force-push, no interactive rebase)
- Commit, rebase, test, and push must be in the **same** iteration

## 3. Update Architecture Docs When You Change Architecture

If you add/remove services, alter data flow, or change module boundaries:
1. Update the relevant section under `docs/kb/`
2. Update its entry in `docs/kb/_index.json`
3. Set `coupled_files` to the source files so future changes trigger review

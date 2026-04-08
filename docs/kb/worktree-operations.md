# Worktree-Aware Operations

## Overview

Agents in this project run in **git worktrees** (`.worktrees/<agent-name>/`).
Each worktree has its own HEAD, index, and working directory but shares the
object store and refs with the main repo. This has practical implications
that agents must account for.

## Key Facts

- **Your cwd is a worktree**, not the main repo root.
  Check `workspace_info.worktree_path` and `workspace_info.main_repo_root`.
- **`git ls-files`** operates on YOUR worktree's index.
- **`git status`** shows YOUR worktree's state.
- **Shared refs**: commits, branches, and tags are shared across all worktrees.
- **Private index**: staging (`git add`) only affects your worktree.

## Cogitor-Managed Files

The cogitor session system writes these files to each worktree root:

| File | Purpose | Tracked? |
|------|---------|----------|
| `.cogitor.heartbeat` | JSON with timestamp, iteration, pid | No (gitignored) |
| `.cogitor.session_bead_id` | Current session bead ID | No (gitignored) |

These files are listed in `.gitignore` under the "Cogitor / worktree artifacts"
section. If they ever appear in `git status` as tracked, it means they were
`git add`-ed before the ignore rules existed. Fix with:

```bash
git rm --cached .cogitor.heartbeat .cogitor.session_bead_id
```

## Common Pitfalls

### 1. Never use `content` edits on files you haven't read

In a worktree you may not have full visibility into existing file contents.
Using a `content` edit (full file replacement) on an existing file will
**destroy** whatever was there. Always use `search_replace` for existing files,
or request the file first with `NEED_FILE:`.

### 2. Always run `git status` before diagnosing git issues

Don't guess filenames. Run `git status` or `git ls-files | grep <pattern>`
to see what's actually tracked, modified, or untracked.

### 3. .gitignore doesn't affect tracked files

`.gitignore` only prevents **untracked** files from being staged. If a file
was committed before the ignore rule was added, the rule has no effect.
Fix: `git rm --cached <file>` to untrack it.

### 4. Merging happens from main repo root

Session completion requires:
```bash
cd /path/to/main/repo    # workspace_info.main_repo_root
git checkout <parent>     # workspace_info.parent_branch
git merge --no-edit agent/<worktree-name>
git push
```

Do NOT push `agent/*` branches to remote.

## Session File Cleanup

When closing a worktree, cogitor should clean up `.cogitor.*` files.
If a session crashes or is abandoned, these files may linger. They are
harmless (gitignored) but can be removed manually:

```bash
rm -f .cogitor.heartbeat .cogitor.session_bead_id
```

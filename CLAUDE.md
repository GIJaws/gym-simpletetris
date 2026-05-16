# CLAUDE.md — gym-simpletetris

> Workspace overview: [`../CLAUDE.md`](../CLAUDE.md). Sibling repo guide (canonical): [`../tetris-ai-models/AGENTS.md`](../tetris-ai-models/AGENTS.md). Sibling Claude notes: [`../tetris-ai-models/CLAUDE.md`](../tetris-ai-models/CLAUDE.md).

Custom fork of `gym_simpletetris`. Used editable by `tetris-ai-models` (the consumer). This repo owns env-side behavior only.

## Defaults

- Default branch: `dev` (not `main`).
- Installed editable from `../tetris-ai-models` via `uv sync` — env changes here are picked up immediately by the models repo on the same machine.

## Platforms

Repo is developed on **Mac** and **Windows desktop**. Worktree directories (`.git/worktrees/...` or `.claude/worktrees/...`) are filesystem-local — branches sync via git, worktree paths don't. To write a file at the repo's canonical location from inside any worktree, use `git rev-parse --show-toplevel`.

## When to work here vs `tetris-ai-models`

Work here only when the issue is env-side:
- observation shape, action mapping, reward shaping, termination, reset/step bugs, game-rule changes.

Anything model/training/eval-side belongs in `tetris-ai-models`. Cross-repo bugs are common — read both repos before assuming ownership.

## Env-side contracts that affect trainers

`tetris-ai-models` enforces these (see its `CLAUDE.md`) — env-side code here must respect them so trainers stay sound:

- If `info["macro_action"]` is emitted, `info["macro_turns"]` and `info["macro_target_x"]` MUST be emitted alongside (co-required bundle).
- `info["shaped_reward"]` is optional; trainers render "n/a" when absent. Don't paper over absence by emitting `0.0`.

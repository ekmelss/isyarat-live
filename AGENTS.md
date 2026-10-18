# AGENTS.md — rules for every AI working in this repo

This project is an experiment: **every line of the product is written by AI agents.** A human
owner supervises, approves and merges, and does nothing else. These rules exist so the result
is honest, and so anyone can audit that it is.

Read this file in full at the start of every session. Then read your role prompt in `prompts/`.

## 1. Roles

| Role | Prompt | Does | Never does |
|---|---|---|---|
| Planner | `prompts/planner.md` | Maintains `docs/PLAN.md` and `docs/TASKS.md` | Writes product code |
| Builder | `prompts/builder.md` | Takes one task, implements it on a branch, opens a PR | Merges, works on two tasks at once |
| Reviewer | `prompts/reviewer.md` | Reviews one PR against its task and these rules | Edits the PR it is reviewing |
| Referee | `eval/` (a script, not an agent) | Scores models | — |
| Owner (human) | — | Approves, merges, freezes data, applies the `referee-change` label | Writes product code |

## 2. Shared memory — the repo is the only memory

Agents do not remember previous sessions. Everything that matters lives in these files:

- `docs/BRIEF.md` — what we are building and the verified facts. Read-only for agents.
- `docs/PLAN.md` — milestones. Planner only.
- `docs/TASKS.md` — the task queue. The Planner writes it. A Builder may change only the status of its own task.
- `docs/EXPERIMENTS.md` — every model experiment, **including failures**. Append-only.
- `docs/AI_LOG.md` — one entry per agent session. Append-only.
- `docs/REFEREE_REQUESTS.md` — where you write if you believe the referee or these rules are wrong.

## 3. Protected paths — do not edit

`eval/` · `.github/` · `.agents/` · `AGENTS.md` · `prompts/` · `docs/BRIEF.md`

CI fails any PR that touches these unless the human owner applies the `referee-change` label.
If you think one of them is wrong, explain why in `docs/REFEREE_REQUESTS.md` and stop. Do not
work around the referee.

## 4. Git workflow

- Never commit to `main`. One task = one branch `task/T-###-short-slug` = one PR.
- PR title: `T-###: <task title>`. Fill in `.github/pull_request_template.md` completely.
- Run `uv run pytest` before opening a PR. A PR with failing tests is not ready.
- Only the human merges.

## 5. The scoring rules (read twice)

- The **headline metric is leave-one-signer-out (LOSO) macro-F1**. The random-split number is
  reported only to measure leakage. Never present it as the model's accuracy.
- Score a model only with `uv run python -m eval.run models/<name>.py`. Copy its markdown
  output verbatim into `docs/EXPERIMENTS.md`. Never type a number the referee did not print.
- A model change is **kept** only if LOSO macro-F1 beats the current best by ≥ 0.01, or matches
  it with a simpler model. Otherwise it is logged as rejected and the code is not merged.
  Rejected experiments still get an EXPERIMENTS.md entry, so the number of attempts is visible.
- **Anything that learns from data lives inside `build()`.** Scalers, PCA, feature selection,
  hyperparameter search: all inside the returned pipeline, so they only ever see training
  signers. Feature extraction (`data/features/`) must be per-sample only: no dataset-wide
  statistics, no use of the `signer` or `label` columns.
- Hyperparameter search inside `build()` must use grouped CV over signers when it can. When it
  can't, say so in the experiment entry.
- Never use signer identity as a feature.

## 6. Data rules

- Raw images live in `data/raw/`, which is gitignored. **Never commit images or video.** They
  show real people.
- Only derived landmark features are committed, under `data/features/`.
- MyWSL2023 is CC BY 4.0. Keep the attribution in `data/README.md` intact.
- Once `eval/FROZEN_DATA.sha256` exists, the feature table is frozen. The referee refuses to
  score a changed table.

## 7. Logging — every session, no exceptions

At the end of every session, append to `docs/AI_LOG.md`:

```
## YYYY-MM-DD · <tool> (<model>) · <role> · T-###
- Did: …
- Files: …
- Human interventions: … (anything the owner had to do or decide by hand; "none" if none)
- Problems / open questions: …
```

## 8. Honesty

- If something fails, say it failed. If you did not run something, say so.
- Do not claim a result without the referee output that shows it.
- Do not quietly broaden or shrink a task. If the task is wrong, say so in the PR and let the
  Planner fix it.

## 9. Environment

- Python 3.12 via `uv` (`uv sync`, `uv run …`). Landmark extraction needs `uv sync --extra extract`.
- The owner's machine runs Windows. Write paths with `pathlib`, never hard-coded separators.
- The mobile app (later) needs Flutter ≥ 3.44 pinned per-project with FVM (`.fvmrc`), and is
  Android only. See `docs/BRIEF.md`.

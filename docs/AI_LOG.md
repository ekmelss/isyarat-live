# AI log

Append-only. One entry per agent session (format in AGENTS.md §7). This file is the evidence
for the claim that the project was built by AI, so record human interventions too.

## 2026-10-18 · Claude Code (Claude Opus 5.5) · scaffolding (outside the agent loop) · —
- Did: set up the repo before the agent loop starts. Wrote the referee (`eval/`) and its 16
  tests, the rules (`AGENTS.md`), the role prompts, `docs/BRIEF.md` (from the owner's earlier
  proposal), the CI guard and the PR template. Inspected MyWSL2023 and found it has no signer labels.
- Files: everything in the initial commit.
- Human interventions: the owner chose the project, the tools (Gemini as the agent loop), and
  public visibility. The owner asked for Claude to stay out of the loop after this setup.
- Problems / open questions: signer labels must be recovered before LOSO can run (BRIEF §4).
  Why the referee was written outside the loop: the agents should not write the code that judges them.
- Same day: Google had ended Gemini CLI sign-in for personal/AI Pro accounts on 2026-06-18, so
  the loop moved to Antigravity CLI (`agy` 1.4.2). Removed `GEMINI.md` (agy loads `AGENTS.md`
  directly), added `/planner` `/builder` `/reviewer` skills under `.agents/skills/` (now protected),
  and set deny rules in the owner's agy settings. Verified live: agy loads AGENTS.md, lists the
  three skills, and is denied both `write_file(eval/metrics.py)` and `gh label list`.
## 2026-10-19 · Antigravity CLI (Gemini 3.8 Flash High) · Planner · —
- Did: Established initial roadmap in `docs/PLAN.md` (milestones M1 to M5) and task queue in `docs/TASKS.md` (tasks T-001 through T-007). Focused M1 on the critical path of recovering MyWSL2023 signer identities with human-auditable contact sheets, extracting per-sample landmarks to `data/features/static.csv`, and preparing for data freezing before baseline model evaluation.
- Files: `docs/PLAN.md`, `docs/TASKS.md`, `docs/AI_LOG.md`.
- Human interventions: none.
- Problems / open questions: Visual verification by the human owner will be required between T-002 and T-004 to confirm the 5 signer labels across the 3,500 images before data is frozen.
## 2026-10-19 · Antigravity CLI (Gemini 3.8 Flash High) · Planner · —
- Did: Refined acceptance criteria in `docs/TASKS.md` for tasks T-001 through T-004 to explicitly specify dataset download targets (`--dataset raw` vs `--dataset crop`), thumbnail downscaling, range definition schema (`signer_ranges.yaml`), and per-signer contact sheet generation (`--by-signer`) for human verification prior to landmark extraction and data freeze. Confirmed T-001 is ready for the first Builder session.
- Files: `docs/TASKS.md`, `docs/AI_LOG.md`.
- Human interventions: none.
- Problems / open questions: none. T-001 has no dependencies and is ready for a Builder.

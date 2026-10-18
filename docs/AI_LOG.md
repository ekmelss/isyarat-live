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

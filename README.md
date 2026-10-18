# Isyarat Live

On-device recognition of **Bahasa Isyarat Malaysia** (Malaysian Sign Language), evaluated the
honest way: on signers the model has never seen.

**This repo is also an experiment: it is built entirely by AI agents.** A human owner supervises
and merges and writes no product code. Every agent session is logged in
[`docs/AI_LOG.md`](docs/AI_LOG.md), and every model attempt is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), including the ones that failed.

## How it is organised

| | |
|---|---|
| [`AGENTS.md`](AGENTS.md) | The rules every agent follows |
| [`prompts/`](prompts/) | Role prompts: Planner, Builder, Reviewer |
| [`docs/BRIEF.md`](docs/BRIEF.md) | What we're building and the verified facts |
| [`docs/PLAN.md`](docs/PLAN.md) · [`docs/TASKS.md`](docs/TASKS.md) | The plan and the task queue |
| [`eval/`](eval/) | **The referee**: a fixed scoring script the agents may run but not edit |

The referee scores each model twice: **leave-one-signer-out** (the headline) and a **random
split** (the leaky comparison most papers report). The gap between them is one of the results.
CI blocks any PR that edits the referee or the rules unless the owner approves it.

## Running

```
uv sync
uv run pytest
uv run python -m eval.run models/<name>.py
```

## Data

Built on **MyWSL2023** (Mendeley Data, DOI 10.17632/zvk55p7ktd.1, CC BY 4.0). Only derived
hand-landmark features are stored here, never images. See [`data/README.md`](data/README.md).

## Status

Scaffolding done. Agent loop not started yet.

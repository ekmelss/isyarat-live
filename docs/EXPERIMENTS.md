# Experiments

Append-only. Every experiment gets an entry, **including rejected ones**: the number of attempts
is part of the honest record.

Template:

```
## E-### · YYYY-MM-DD · models/<file>.py · T-### · KEPT | REJECTED
Hypothesis: …
Change vs current best: …
Referee output (verbatim from `uv run python -m eval.run …`):
<paste>
Decision: … (current best LOSO macro-F1 was X.XXX)
```

**Current best:** none yet.

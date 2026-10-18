# Plan

## Milestones

### M1: Signer Recovery & Data Freeze
- **Goal:** Recover signer identities across all 3,500 MyWSL2023 frames, visually verify them via contact sheets, extract per-sample MediaPipe landmarks into `data/features/static.csv`, and have the owner freeze the dataset via `eval/freeze`.
- **Checkable condition:** `eval/FROZEN_DATA.sha256` exists and `data/features/static.csv` passes `eval.data.load()`.

### M2: Baseline Model & Leakage Finding
- **Goal:** Build the first static-sign classifier in `models/`, score it with the referee (`eval.run`), record the first entry in `docs/EXPERIMENTS.md`, and document the leakage gap between leave-one-signer-out (LOSO) and random split.
- **Checkable condition:** `docs/EXPERIMENTS.md` contains the first verified referee run; baseline LOSO macro-F1 and leakage gap are established.

### M3: Model Improvement & Confidence Gating
- **Goal:** Explore normalisation techniques and model architectures (within `build()`), evaluate calibration (ECE), and implement an abstain gate that refuses low-confidence predictions.
- **Checkable condition:** An improved model beats baseline by >= 0.01 LOSO macro-F1 or matches with a simpler pipeline; abstain thresholds are calibrated and documented.

### M4: Android App (Flutter)
- **Goal:** On-device Android application using Flutter (pinned via FVM), live camera feed, MediaPipe hand landmarker, offline classification, confidence gate, and latency benchmarking.
- **Checkable condition:** Working Android build running inference offline at < 50ms/frame on device.

### M5: Web Dashboard & Audit Artifact
- **Goal:** Next.js static web dashboard displaying BIM vocabulary, interactive confusion matrices, per-signer performance breakdown, and the side-by-side leakage demonstration.
- **Checkable condition:** Deployable dashboard showcasing the project's honest evaluation findings and vocabulary guide.

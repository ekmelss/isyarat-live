# Tasks

*Owned by the Planner. A Builder changes only the status of its own task.*

Format:

```
### T-### — <title>
- Status: todo | doing | review | done | dropped
- Depends on: none | T-###
- Acceptance criteria:
  - …
- Out of scope:
  - …
```

<!-- Tasks go below this line. -->

### T-001 — Dataset download and verification helper
- Status: review
- Depends on: none
- Acceptance criteria:
  - `tools/download_data.py` exists and is runnable via `uv run python -m tools.download_data`.
  - Downloads MyWSL2023 dataset from Mendeley Data DOI `10.17632/zvk55p7ktd.1` into `data/raw/` (gitignored). Supports `--dataset raw` (default, 3.3 GB full frames) and `--dataset crop` (54 MB tight hand crops).
  - Verifies archive integrity and directory structure after download.
  - Supports a `--check` flag to verify local archive/extracted file integrity without re-downloading.
  - Unit tests in `tests/test_download.py` test URL handling, extraction logic, and checksum verification with mocks.
  - `uv run pytest` passes.
  - `git status` confirms no images or zip files under `data/raw/` are tracked by git.
- Out of scope:
  - Landmark extraction.
  - Modifying files under `eval/` or any protected paths.
  - Committing any image or binary files.

### T-002 — Contact sheet generator for visual signer verification
- Status: todo
- Depends on: T-001
- Acceptance criteria:
  - `tools/generate_contact_sheets.py` exists and is runnable via `uv run python -m tools.generate_contact_sheets`.
  - Generates HTML contact sheets in `data/raw/sheets/` (gitignored) for all 10 words, displaying lightweight downscaled thumbnails with image sequence numbers to identify signer runs.
  - Supports a `--by-signer <annotations_csv>` flag to generate per-signer contact sheets (`signer_S1.html` .. `signer_S5.html`) once annotations exist.
  - Preserves face pixelation from raw images while making clothing/sleeves/hands clearly visible for human signer identification.
  - Unit tests in `tests/test_contact_sheets.py` test grid generation and HTML rendering using synthetic mock images.
  - `uv run pytest` passes.
  - No generated contact sheets or thumbnails are committed to git.
- Out of scope:
  - The actual manual visual auditing (owner responsibility).
  - MediaPipe landmark extraction.
  - Writing `data/features/static.csv`.

### T-003 — Automated signer run proposal tool
- Status: todo
- Depends on: T-002
- Acceptance criteria:
  - `tools/propose_signer_runs.py` exists and is runnable via `uv run python -m tools.propose_signer_runs`.
  - Analyzes image sequences for all 10 words to propose signer runs automatically (e.g. via clothing/torso/sleeve color clustering or sharp sequence change-point detection across the 350 frames per word).
  - Emits draft `data/annotations/signer_ranges.yaml` containing proposed run intervals mapped to signer IDs (`S1`..`S5`), a confidence score (0.0 to 1.0) per run, and flags runs with low confidence or ambiguous transitions (`needs_human_check: true`).
  - Incorporates known run anchor cues from `docs/BRIEF.md` §4 to bootstrap initial signer mappings.
  - Prints a concise summary table indicating proposed runs, confidence scores, and items flagged for human verification.
  - Unit tests in `tests/test_propose_runs.py` test run detection and YAML output formatting using synthetic mock frame sequences (e.g. solid-color blocks simulating distinct clothing).
  - `uv run pytest` passes.
- Out of scope:
  - Manual verification on contact sheets (owner responsibility).
  - Landmark extraction.
  - Modifying `eval/`.

### T-004 — Signer annotation schema, validator, and expansion
- Status: todo
- Depends on: T-003
- Acceptance criteria:
  - Signer annotation range definition schema established in `data/annotations/signer_ranges.yaml` mapping contiguous image index intervals per word to signers `S1`..`S5` (confirmed and corrected by owner after T-003).
  - `tools/validate_signers.py` exists and is runnable via `uv run python -m tools.validate_signers <path_to_annotations.yaml>` with `--expand` flag to generate the full 3,500-sample mapping in `data/annotations/signers.csv`.
  - Validation checks that all 3,500 samples across the 10 words are mapped, exactly 5 signers are present, each signer signs all 10 words, and there are no duplicate or missing indices or overlapping ranges.
  - Generates the 5 per-signer verification sheets using `tools/generate_contact_sheets.py --by-signer` for the human owner to inspect before freezing.
  - Unit tests in `tests/test_validate_signers.py` test valid mapping as well as failure modes (missing frame, unknown signer id, missing word, overlapping ranges).
  - `uv run pytest` passes.
- Out of scope:
  - Modifying `eval/`.
  - Feature extraction.

### T-005 — MediaPipe landmark extraction pipeline
- Status: todo
- Depends on: T-004
- Acceptance criteria:
  - `tools/extract_landmarks.py` exists and is runnable via `uv run python -m tools.extract_landmarks`.
  - Reads verified signer labels from `data/annotations/signers.csv` to populate the `signer` column.
  - Uses MediaPipe Hand Landmarker (`uv sync --extra extract`) to extract 21 landmarks (x, y, z) per detected hand.
  - Normalisation: raw landmark coordinates only, extracted strictly per-sample (no dataset-wide statistics).
  - Writes `data/features/static.csv` matching `eval/data.py` required columns (`sample_id`, `signer`, `label`, `f0`..`fN`).
  - Writes `data/features/EXTRACTION_REPORT.md` reporting detection counts and drop rates (frames with no hand detected) per word and per signer.
  - Unit tests in `tests/test_extraction.py` test landmark formatting, column ordering, and handling of undetected hands.
  - `uv run pytest` passes.
- Out of scope:
  - Landmark normalisation (wrist-centering, scale normalisation belongs in `models/<name>.py:build()`).
  - Running `eval.freeze` (owner action).

### T-006 — Dataset validation against referee loader
- Status: todo
- Depends on: T-005
- Acceptance criteria:
  - `tests/test_data_integrity.py` calls `eval.data.load("data/features/static.csv")` and asserts that the dataset loads without `DataError`.
  - Asserts that all 10 labels exist, >= 2 signers exist, all feature values are finite numbers, and no NaNs exist.
  - Prints clear verification status and instructions for the owner to freeze data via `uv run python -m eval.freeze`.
  - `uv run pytest` passes.
- Out of scope:
  - Running `eval.freeze` or modifying `eval/` (owner only via `referee-change`).
  - Training models.

### T-007 — Baseline static sign model (Logistic Regression)
- Status: todo
- Depends on: T-006
- Acceptance criteria:
  - `models/baseline_logreg.py` exists, exposes `build()` returning an unfitted scikit-learn Pipeline, and defines `DESCRIPTION`.
  - All feature normalisation (wrist-centering on landmark 0, scale normalisation by hand span) is encapsulated inside `build()`.
  - Model implements `fit`, `predict`, and `predict_proba` with `classes_`.
  - Scored with the referee: `uv run python -m eval.run models/baseline_logreg.py`.
  - Referee output is copied verbatim into `docs/EXPERIMENTS.md` as experiment `E-001`, documenting baseline LOSO macro-F1 and the leakage gap.
  - Unit tests in `tests/test_models.py` verify that `build()` returns a valid scikit-learn estimator compatible with `eval.protocols.run`.
  - `uv run pytest` passes.
- Out of scope:
  - Any preprocessing or scaling outside of `build()`.
  - Touching files in `eval/`.

### T-008 — Calibration analysis and confidence gating
- Status: todo
- Depends on: T-007
- Acceptance criteria:
  - `tools/calibrate.py` exists and is runnable via `uv run python -m tools.calibrate models/baseline_logreg.py`.
  - Computes Expected Calibration Error (ECE) and evaluates accuracy vs coverage at varying confidence thresholds tau.
  - Generates reliability diagram and coverage curve data.
  - Unit tests in `tests/test_calibration.py` test confidence gating logic and threshold calculation.
  - `uv run pytest` passes.
- Out of scope:
  - Modifying `eval/metrics.py`.

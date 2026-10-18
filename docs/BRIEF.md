# Brief — Isyarat Live

*Read-only for agents. The owner changes it through a `referee-change` PR.*

## 1. Goal

Recognise **Bahasa Isyarat Malaysia (BIM)** signs from a phone camera, **entirely on-device**,
and measure honestly how well it works for a person the model has never seen.

Pipeline: camera frame → MediaPipe Hand Landmarker (21 landmarks × x, y, z per hand) → per-sample
geometric normalisation → small classifier → confidence gate that abstains rather than guessing.

## 2. Deliverables, in order

1. **Static-sign classifier** over hand landmarks, scored by the referee (`eval/`).
2. **The leakage finding**: LOSO vs random split on identical data and model, side by side.
3. **Calibration + confidence gate**: a reliability diagram, and an abstain threshold.
4. **Android app** (Flutter): live recognition, offline.
5. **Web dashboard** (Next.js): vocabulary, per-class metrics, confusion matrix.
6. Later, optional: dynamic (moving) signs.

Items 1–3 come first. Do not start 4–5 before 1 has an honest baseline.

## 3. The core idea

When a dataset has few signers and is split randomly, the same person's hands land in both train
and test, so the score measures memorising people. This project reports **leave-one-signer-out**
as the headline and the random split only as the comparison. The gap between them is a result,
not a bug. Expect the honest number to be **lower** than published BIM papers.

Prior BIM recognition exists (pixel-based MobileNet on fingerspelling; SSD-MobileNet-V2 FPNLite
in MDPI *Information* 14(6):319). **Never claim this is the first BIM recogniser.**

## 4. Dataset: MyWSL2023 — facts verified 2026-10-18

- Mendeley Data, DOI `10.17632/zvk55p7ktd.1`, **CC BY 4.0**. Paper: *Data in Brief*,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10439288/
- Two zips:
  - `MyWSL2023 CROP DATA.zip`: **54 MB**, tight hand crops (~200×370 px, often no wrist).
  - `MyWSL2023 RAW DATA.zip`: **3.3 GB**, 1920×1080 full frames, faces pixelated by the authors.
- Both have the same layout: `train/<word>/<word> (N).jpg` for N = 1–300 and
  `test/<word>/<word> (N).jpg` for N = 301–350. 350 images × 10 words = 3,500.
- Folder names are Malay: `air` (water), `demam` (fever), `dengar` (hear), `makan` (eat),
  `minum` (drink), `salah` (wrong), `saya` (I), `senyap` (quiet), `tidur` (sleep), `waktu` (time).
- The paper reports **5 signers (2 male, 3 female, aged 20–21)**.
- ⚠️ **There are NO signer labels.** Not in file names, not in folders, not in the paper.
  A spot check of `air` raw frames showed contiguous runs per signer, identifiable by clothing:
  N=1 man in a red polo · N=70, 71, 140, 141 man in a black T-shirt · N=210, 211 woman in a
  brown hijab · N=280, 281 woman in a black hijab · N=301 man in a red jersey. Run boundaries
  are **not** multiples of 70. Whether the two red shirts are the same man is **unknown**, and
  so is how the runs fall in the other nine words.
- The author `train`/`test` folders are an index split (301–350), not a signer split. Ignore
  them for evaluation, but record which signer(s) fall in 301–350. That is interesting.
- Hand detection on the tight crops may fail often. The raw frames are the safer source for
  landmarks, but mind the 3.3 GB download. Zip entries can be fetched individually with HTTP
  range requests if needed.

**Consequence:** recovering one signer id per image, verified by the owner by eye, is the first
critical-path work. Until then LOSO cannot run.

Other BIM corpora (BIM-SSD: needs ethics approval; BIM-SIGN: waitlisted) are upside, never
dependencies.

## 5. Evaluation (enforced by `eval/`)

- Headline: LOSO macro-F1, mean ± std across signers, plus the worst signer.
- Comparison: stratified 5-fold random split on the same data and the same `build()`.
- Also: accuracy, pooled confusion matrix, expected calibration error (when the model has
  `predict_proba`).
- On-device latency (later): ms per frame on a named Android handset.

## 6. Constraints

- Mobile is **Android only**: the Flutter `hand_landmarker` plugin (v3.1.0) has no iOS support.
  It needs Flutter ≥ 3.44 and JDK ≥ 17. Pin Flutter per-project with FVM.
- Zero cost: free tiers only. The model is tiny, and CPU training is enough.
- Static signs only for now. No facial or non-manual features.

## 7. Ethics

- Raw images show real people. Never commit them, and never publish crops of faces.
- Publish landmark features only.
- The app must say plainly that it is an assistive aid, not an interpreter.
- If new signers are ever recorded: informed consent first, covering whether clips may be published.

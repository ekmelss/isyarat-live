# Data

## Source and attribution

**MyWSL2023: Malaysian Words Sign Language dataset.** Mendeley Data, DOI
[10.17632/zvk55p7ktd.1](https://doi.org/10.17632/zvk55p7ktd.1). Licensed **CC BY 4.0**.
Described in *Data in Brief*: https://pmc.ncbi.nlm.nih.gov/articles/PMC10439288/

The features in `data/features/` are derived from that dataset (hand landmarks extracted with
MediaPipe). No images are redistributed here.

## Layout

```
data/raw/        gitignored: downloaded zips and extracted images. Never commit.
data/features/   committed: derived, anonymous landmark tables
```

## The table the referee scores: `data/features/static.csv`

| Column | Meaning |
|---|---|
| `sample_id` | Unique, stable id, e.g. `air_0001` |
| `signer` | Signer id, e.g. `S1`…`S5`. Recovered by annotation (see docs/BRIEF.md §4) |
| `label` | The word, Malay folder name (`air`, `demam`, …) |
| `f0` … `fN` | Raw landmark coordinates as MediaPipe returns them, in a documented order |

Extraction must be **per-sample**: no dataset-wide statistics. Normalisation belongs in the
model's `build()`. Samples where no hand is detected are dropped, and the count is reported per
word and per signer in `data/features/EXTRACTION_REPORT.md`.

## Freezing

Once the owner has checked the signer labels by eye and accepted the table, the owner runs:

```
uv run python -m eval.freeze
```

and commits it in a `referee-change` PR. From then on the referee refuses to score a changed table.

"""Loads and validates the feature table the referee scores against.

Part of the referee. Agents must not edit anything under eval/ (see AGENTS.md).
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "signer", "label")
FREEZE_FILE = Path(__file__).with_name("FROZEN_DATA.sha256")


class DataError(ValueError):
    """The feature table cannot be scored as it stands."""


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    signers: np.ndarray
    sample_ids: np.ndarray
    feature_names: list
    digest: str
    frozen: bool
    warnings: list = field(default_factory=list)


def digest_of(path):
    """sha256 of the file with line endings normalised, so Windows and CI agree."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _feature_columns(columns):
    feats = [c for c in columns if c.startswith("f") and c[1:].isdigit()]
    return sorted(feats, key=lambda c: int(c[1:]))


def load(path, freeze_file=FREEZE_FILE):
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} does not exist. See data/README.md for how it is produced.")

    digest = digest_of(path)
    frozen = Path(freeze_file).exists()
    if frozen:
        expected = Path(freeze_file).read_text().split()[0].strip()
        if digest != expected:
            raise DataError(
                f"{path} has changed since it was frozen "
                f"(expected sha256 {expected[:12]}…, got {digest[:12]}…). "
                "Frozen data may only change through a human-approved referee-change PR."
            )

    df = pd.read_csv(path, dtype={"sample_id": str, "signer": str, "label": str})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"missing required column(s): {', '.join(missing)}")

    feats = _feature_columns(df.columns)
    if not feats:
        raise DataError("no feature columns found (expected f0, f1, …)")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any() or (df[col].str.strip() == "").any():
            raise DataError(f"column '{col}' has empty values")

    dupes = df["sample_id"][df["sample_id"].duplicated()].unique()
    if len(dupes):
        raise DataError(f"duplicate sample_id(s): {', '.join(map(str, dupes[:5]))}")

    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X).all():
        bad = df["sample_id"][~np.isfinite(X).all(axis=1)].head(5).tolist()
        raise DataError(f"non-numeric, NaN or infinite feature values, e.g. sample(s) {bad}")

    signers = df["signer"].to_numpy()
    y = df["label"].to_numpy()
    if len(np.unique(signers)) < 2:
        raise DataError("need at least 2 signers for leave-one-signer-out")
    if len(np.unique(y)) < 2:
        raise DataError("need at least 2 labels")

    warnings = []
    if not frozen:
        warnings.append("data is NOT frozen yet; scores are provisional (see data/README.md)")
    all_labels = set(y)
    for s in sorted(np.unique(signers)):
        lacking = all_labels - set(y[signers == s])
        if lacking:
            warnings.append(f"signer {s} has no samples for: {', '.join(sorted(lacking))}")

    return Dataset(
        X=X,
        y=y,
        signers=signers,
        sample_ids=df["sample_id"].to_numpy(),
        feature_names=feats,
        digest=digest,
        frozen=frozen,
        warnings=warnings,
    )

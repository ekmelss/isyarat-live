"""The two evaluation protocols: leave-one-signer-out (headline) and random split (leakage).

Part of the referee. Agents must not edit anything under eval/ (see AGENTS.md).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

RANDOM_SPLIT_FOLDS = 5
RANDOM_SPLIT_SEED = 0


@dataclass
class FoldResult:
    name: str
    test_index: np.ndarray
    y_pred: np.ndarray
    proba: Optional[np.ndarray]
    classes: Optional[np.ndarray]
    fit_seconds: float


def loso_folds(signers):
    """One fold per signer: train on everyone else, test on that signer."""
    for s in sorted(np.unique(signers)):
        test = np.flatnonzero(signers == s)
        train = np.flatnonzero(signers != s)
        # The guarantee the whole project rests on. Never remove this.
        assert not set(signers[train]) & set(signers[test]), "signer leaked across split"
        yield f"signer={s}", train, test


def random_folds(y, n_splits=RANDOM_SPLIT_FOLDS, seed=RANDOM_SPLIT_SEED):
    """Stratified random folds that ignore signers. Leaky on purpose: it is the comparison."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        yield f"random-{i}", train, test


def run(build, X, y, folds):
    """Fit a fresh model per fold. build() is called every time so no state crosses folds."""
    import time

    results = []
    for name, train, test in folds:
        model = build()
        start = time.perf_counter()
        model.fit(X[train], y[train])
        fit_seconds = time.perf_counter() - start
        y_pred = np.asarray(model.predict(X[test]))
        proba, classes = None, None
        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X[test]))
                classes = np.asarray(model.classes_)
            except (AttributeError, NotImplementedError):
                proba, classes = None, None
        results.append(FoldResult(name, test, y_pred, proba, classes, fit_seconds))
    return results

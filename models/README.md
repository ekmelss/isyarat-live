# Models

One file per candidate. The referee imports the file and calls `build()` once per fold.

```python
DESCRIPTION = "Logistic regression on wrist-centred, span-scaled landmarks"

def build():
    """Return a NEW, unfitted scikit-learn-style estimator.

    It receives raw feature rows (the f0…fN columns of data/features/static.csv) and string
    labels. Anything that learns from data (scaling, PCA, tuning) must sit inside the returned
    pipeline. Set random_state on anything random.
    """
    ...
```

Required: `fit(X, y)` and `predict(X)`. Optional: `predict_proba(X)` with `classes_`, which
enables the calibration score.

Score it with `uv run python -m eval.run models/<file>.py`.

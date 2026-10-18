"""Tests that the referee measures what it claims to. Run: uv run pytest"""

import json

import numpy as np
import pandas as pd
import pytest

from eval import data, metrics, protocols, run

MEMORISER = """
from sklearn.neighbors import KNeighborsClassifier
DESCRIPTION = "1-NN, memorises individuals"
def build():
    return KNeighborsClassifier(n_neighbors=1)
"""


def synthetic_csv(path, n_signers=5, n_labels=4, per_cell=20, signer_shift=6.0, class_shift=1.0, seed=0):
    """Each signer gets a large private offset; classes differ only slightly.

    A model that memorises signers looks great on a random split and poor on LOSO,
    which is exactly the leakage the referee exists to expose.
    """
    rng = np.random.default_rng(seed)
    rows = []
    signer_offsets = rng.normal(0, signer_shift, size=(n_signers, 8))
    class_offsets = rng.normal(0, class_shift, size=(n_labels, 8))
    i = 0
    for s in range(n_signers):
        # A signer-specific twist on each class, so learning a class from other signers transfers poorly.
        twist = rng.normal(0, class_shift * 2, size=(n_labels, 8))
        for c in range(n_labels):
            for _ in range(per_cell):
                x = signer_offsets[s] + class_offsets[c] + twist[c] + rng.normal(0, 0.3, 8)
                rows.append({"sample_id": f"s{i}", "signer": f"S{s}", "label": f"L{c}",
                             **{f"f{k}": v for k, v in enumerate(x)}})
                i += 1
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def csv(tmp_path):
    return synthetic_csv(tmp_path / "static.csv")


@pytest.fixture
def no_freeze(tmp_path):
    return tmp_path / "absent.sha256"


@pytest.fixture
def model_file(tmp_path):
    p = tmp_path / "memoriser.py"
    p.write_text(MEMORISER)
    return p


def test_loso_never_shares_a_signer_between_train_and_test():
    signers = np.array(list("AABBBCCDDDDE"))
    seen = []
    for name, train, test in protocols.loso_folds(signers):
        assert not set(signers[train]) & set(signers[test])
        assert len(set(signers[test])) == 1
        assert len(train) + len(test) == len(signers)
        seen.extend(test)
    assert sorted(seen) == list(range(len(signers)))


def test_referee_exposes_a_model_that_memorises_signers(csv, model_file, no_freeze):
    report = run.evaluate(model_file, csv, freeze_file=no_freeze)
    assert report["random_split"]["macro_f1_mean"] > 0.9
    assert report["loso"]["macro_f1_mean"] < 0.6
    assert report["leakage_gap"] > 0.3
    assert len(report["loso"]["folds"]) == 5


def test_build_is_called_fresh_for_every_fold(csv, tmp_path, no_freeze):
    counter = tmp_path / "count.txt"
    model = tmp_path / "counting.py"
    model.write_text(
        "from pathlib import Path\n"
        "from sklearn.dummy import DummyClassifier\n"
        f"P = Path({str(counter)!r})\n"
        "def build():\n"
        "    P.write_text(str(int(P.read_text() or 0) + 1) if P.exists() else '1')\n"
        "    return DummyClassifier()\n"
    )
    run.evaluate(model, csv, freeze_file=no_freeze)
    assert int(counter.read_text()) == 5 + protocols.RANDOM_SPLIT_FOLDS


def test_frozen_data_that_changed_is_refused(csv, tmp_path):
    freeze = tmp_path / "FROZEN.sha256"
    freeze.write_text(data.digest_of(csv) + "\n")
    assert data.load(csv, freeze_file=freeze).frozen

    df = pd.read_csv(csv)
    df.loc[0, "f0"] += 1.0
    df.to_csv(csv, index=False)
    with pytest.raises(data.DataError, match="changed since it was frozen"):
        data.load(csv, freeze_file=freeze)


def test_digest_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\n1,2\n")
    b.write_bytes(b"x,y\r\n1,2\r\n")
    assert data.digest_of(a) == data.digest_of(b)


@pytest.mark.parametrize("mutate, message", [
    (lambda df: df.drop(columns=["signer"]), "missing required column"),
    (lambda df: df.assign(sample_id="dup"), "duplicate sample_id"),
    (lambda df: df.assign(f0=np.nan), "non-numeric, NaN or infinite"),
    (lambda df: df.assign(signer="only-one"), "at least 2 signers"),
    (lambda df: df[["sample_id", "signer", "label"]], "no feature columns"),
])
def test_bad_tables_are_refused(csv, no_freeze, mutate, message):
    mutate(pd.read_csv(csv)).to_csv(csv, index=False)
    with pytest.raises(data.DataError, match=message):
        data.load(csv, freeze_file=no_freeze)


def test_feature_columns_are_ordered_numerically(tmp_path, no_freeze):
    p = tmp_path / "t.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "signer": ["S1", "S2"], "label": ["x", "y"],
                  "f10": [10, 10], "f2": [2, 2], "f1": [1, 1]}).to_csv(p, index=False)
    ds = data.load(p, freeze_file=no_freeze)
    assert ds.feature_names == ["f1", "f2", "f10"]
    assert ds.X[0].tolist() == [1, 2, 10]


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.full(10, 0.8)
    correct = np.array([1] * 8 + [0] * 2)
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_ece_catches_overconfidence():
    conf = np.full(10, 1.0)
    correct = np.array([1] * 5 + [0] * 5)
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.5)


def test_macro_f1_counts_a_never_predicted_class_as_zero():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "a", "a", "a"])
    assert metrics.macro_f1(y_true, y_pred, ["a", "b"]) == pytest.approx((2 / 3 + 0) / 2)


def test_cli_writes_json_and_markdown(csv, model_file, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "FREEZE_FILE", tmp_path / "absent")
    out, summary = tmp_path / "results", tmp_path / "summary.md"
    code = run.main([str(model_file), "--data", str(csv), "--out", str(out), "--summary", str(summary)])
    assert code == 0
    report = json.loads((out / "memoriser.json").read_text())
    assert {"loso", "random_split", "leakage_gap", "data_sha256"} <= report.keys()
    assert "Leave-one-signer-out" in summary.read_text(encoding="utf-8")


def test_cli_refuses_missing_data(model_file, tmp_path, capsys):
    assert run.main([str(model_file), "--data", str(tmp_path / "nope.csv")]) == 2
    assert "REFEREE REFUSED" in capsys.readouterr().err

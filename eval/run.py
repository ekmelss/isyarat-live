"""Score a model file under both protocols.

    uv run python -m eval.run models/<name>.py

Part of the referee. Agents must not edit anything under eval/ (see AGENTS.md).
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from eval import data as data_mod
from eval import metrics, protocols

DEFAULT_DATA = Path("data/features/static.csv")
DEFAULT_OUT = Path("results")


def load_model_module(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise SystemExit(f"{path} must define build() returning a fresh estimator (see models/README.md)")
    return module


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def summarise(fold_results, ds, labels):
    folds = []
    for r in fold_results:
        y_true = ds.y[r.test_index]
        folds.append({
            "fold": r.name,
            "n": int(len(r.test_index)),
            "accuracy": metrics.accuracy(y_true, r.y_pred),
            "macro_f1": metrics.macro_f1(y_true, r.y_pred, labels),
        })
    f1s = np.array([f["macro_f1"] for f in folds])
    accs = np.array([f["accuracy"] for f in folds])

    y_true_all = np.concatenate([ds.y[r.test_index] for r in fold_results])
    y_pred_all = np.concatenate([r.y_pred for r in fold_results])

    ece = None
    if all(r.proba is not None for r in fold_results):
        conf, correct = zip(*(metrics.top_confidence(r.proba, r.classes, ds.y[r.test_index])
                              for r in fold_results))
        ece = metrics.expected_calibration_error(np.concatenate(conf), np.concatenate(correct))

    return {
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std()),
        "macro_f1_worst": float(f1s.min()),
        "accuracy_mean": float(accs.mean()),
        "ece": ece,
        "fit_seconds_total": float(sum(r.fit_seconds for r in fold_results)),
        "folds": folds,
        "confusion_pooled": metrics.confusion(y_true_all, y_pred_all, labels),
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def to_markdown(report):
    loso, rnd = report["loso"], report["random_split"]
    worst = min(loso["folds"], key=lambda f: f["macro_f1"])
    lines = [
        f"### Referee: `{report['model']}`",
        "",
        f"**Headline: leave-one-signer-out macro-F1 = {loso['macro_f1_mean']:.3f} "
        f"± {loso['macro_f1_std']:.3f}** (worst: {worst['fold']} at {worst['macro_f1']:.3f})",
        "",
        "| Protocol | Macro-F1 | Accuracy | ECE |",
        "|---|---|---|---|",
        f"| Leave-one-signer-out | {loso['macro_f1_mean']:.3f} | {loso['accuracy_mean']:.3f} | {_fmt(loso['ece'])} |",
        f"| Random split (leaky) | {rnd['macro_f1_mean']:.3f} | {rnd['accuracy_mean']:.3f} | {_fmt(rnd['ece'])} |",
        "",
        f"Leakage gap (random − LOSO macro-F1): **{report['leakage_gap']:+.3f}**",
        "",
        "| Held-out signer | n | Macro-F1 | Accuracy |",
        "|---|---|---|---|",
    ]
    lines += [f"| {f['fold']} | {f['n']} | {f['macro_f1']:.3f} | {f['accuracy']:.3f} |" for f in loso["folds"]]
    lines += [
        "",
        f"Data sha256 `{report['data_sha256'][:12]}` ({'frozen' if report['data_frozen'] else 'NOT frozen'}) · "
        f"model sha256 `{report['model_sha256'][:12]}` · commit `{report['git_commit']}`",
    ]
    lines += [f"> ⚠️ {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def evaluate(model_path, data_path=DEFAULT_DATA, freeze_file=data_mod.FREEZE_FILE):
    ds = data_mod.load(data_path, freeze_file=freeze_file)
    module = load_model_module(model_path)
    labels = sorted(np.unique(ds.y))

    loso = protocols.run(module.build, ds.X, ds.y, protocols.loso_folds(ds.signers))
    rnd = protocols.run(module.build, ds.X, ds.y, protocols.random_folds(ds.y))

    report = {
        "model": Path(model_path).as_posix(),
        "description": getattr(module, "DESCRIPTION", ""),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "data": Path(data_path).as_posix(),
        "data_sha256": ds.digest,
        "data_frozen": ds.frozen,
        "git_commit": git_commit(),
        "n_samples": int(len(ds.y)),
        "n_features": len(ds.feature_names),
        "signers": sorted(map(str, np.unique(ds.signers))),
        "labels": labels,
        "loso": summarise(loso, ds, labels),
        "random_split": summarise(rnd, ds, labels),
        "warnings": ds.warnings,
    }
    report["leakage_gap"] = report["random_split"]["macro_f1_mean"] - report["loso"]["macro_f1_mean"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="path to a models/*.py file defining build()")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the JSON report")
    parser.add_argument("--summary", help="also append the markdown report to this file")
    args = parser.parse_args(argv)

    try:
        report = evaluate(args.model, args.data)
    except data_mod.DataError as e:
        print(f"REFEREE REFUSED: {e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args.model).stem}.json").write_text(json.dumps(report, indent=2))

    md = to_markdown(report)
    print(md)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

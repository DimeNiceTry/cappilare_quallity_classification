"""Evaluate held-out 3-class logits, optionally temperature-scaled on separate patients.

CSV columns: file,patient_id,true_label,logit_not_ok,logit_doubtful,logit_ok
true_label must be not_ok, doubtful, or ok. Optional annotator is preserved in reports.
This script cannot verify that the model did not train on these patients.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

CLASSES = ("not_ok", "doubtful", "ok")


def validate_probabilities(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if p.ndim != 2 or not len(p) or p.shape[1] < 2:
        raise ValueError("Probabilities must have nonempty shape (N, K), K >= 2")
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be an integer vector of length N")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Label index outside probability columns")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Probability rows must sum to 1")
    return y, p


def calibration_bins(confidence, outcome, n_bins=10):
    """Equal-width bins [left, right), with the last bin including 1."""
    confidence, outcome = np.asarray(confidence, float), np.asarray(outcome, float)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if confidence.ndim != 1 or not len(confidence) or outcome.shape != confidence.shape:
        raise ValueError("Nonempty confidence and outcome vectors must match")
    if (not np.all(np.isfinite(confidence)) or np.any(confidence < 0)
            or np.any(confidence > 1) or not np.all(np.isin(outcome, [0, 1]))):
        raise ValueError("Confidence must lie in [0,1] and outcomes must be 0/1")
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        mean_p = float(confidence[mask].mean()) if count else None
        rate = float(outcome[mask].mean()) if count else None
        if count:
            ece += count / len(confidence) * abs(mean_p - rate)
        bins.append(dict(left=float(edges[b]), right=float(edges[b + 1]), count=count,
                         mean_probability=mean_p, observed_frequency=rate))
    return float(ece), bins


def top_label_ece(y, p, n_bins=10):
    y, p = validate_probabilities(y, p)
    return calibration_bins(p.max(axis=1), p.argmax(axis=1) == y, n_bins)[0]


def nll_from_logits(y, logits, temperature=1.0):
    z = np.asarray(logits, float) / temperature
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(z)), y]))


def fit_temperature(y, logits):
    logits = np.asarray(logits, float)
    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must be finite")
    validate_probabilities(y, softmax(logits, axis=1))
    bounds = (float(np.log(.05)), float(np.log(20.0)))
    result = minimize_scalar(lambda t: nll_from_logits(y, logits, np.exp(t)),
                             bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization did not converge")
    # Include T=1 and bounds so calibration NLL cannot increase numerically.
    candidates = [1.0, .05, 20.0, float(np.exp(result.x))]
    temperature = min(candidates, key=lambda t: nll_from_logits(y, logits, t))
    return temperature


def evaluate(y, p, threshold=None):
    y, p = validate_probabilities(y, p)
    pred = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(3), zero_division=0)
    result = {
        "n": len(y), "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, labels=range(3), average="macro", zero_division=0)),
        "confusion_matrix_rows_true_cols_predicted": confusion_matrix(y, pred, labels=range(3)).tolist(),
        "nll_clipped_at_1e-15": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "brier_multiclass_sum_over_classes": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        "top_label_ece": {str(b): top_label_ece(y, p, b) for b in (5, 10, 15)},
        "classwise_ece_10": {c: calibration_bins(p[:, k], y == k, 10)[0] for k, c in enumerate(CLASSES)},
        "per_class": {c: dict(precision=float(precision[k]), recall=float(recall[k]),
                              f1=float(f1[k]), support=int(support[k])) for k, c in enumerate(CLASSES)},
    }
    good = y == 2
    result["ok_vs_rest_auroc"] = float(roc_auc_score(good, p[:, 2])) if len(np.unique(good)) == 2 else None
    result["ok_vs_rest_average_precision"] = float(average_precision_score(good, p[:, 2])) if len(np.unique(good)) == 2 else None
    if threshold is not None:
        accepted = p[:, 2] >= threshold
        count = int(accepted.sum())
        bad = y == 0
        result["acceptance"] = {
            "threshold": threshold, "count": count, "coverage": float(accepted.mean()),
            "precision_ok": float(good[accepted].mean()) if count else None,
            "non_ok_fraction_among_accepted": float((~good)[accepted].mean()) if count else None,
            "recall_ok": float(accepted[good].mean()) if good.any() else None,
            "not_ok_accepted_over_all_not_ok": float(accepted[bad].mean()) if bad.any() else None,
            "doubtful_accepted_count": int(np.sum(accepted & (y == 1))),
        }
    return result


def bootstrap_calibration(y, p, patients, repetitions=1000):
    """Conditional on fixed predictions/calibrator; resample whole patient clusters."""
    patients = np.asarray(patients)
    groups = np.unique(patients)
    indices = {g: np.flatnonzero(patients == g) for g in groups}
    rng = np.random.default_rng(20261101)
    values = []
    for _ in range(repetitions):
        idx = np.concatenate([indices[g] for g in rng.choice(groups, len(groups), replace=True)])
        values.append([top_label_ece(y[idx], p[idx], 10),
                       calibration_bins(p[idx, 2], y[idx] == 2, 10)[0]])
    intervals = np.percentile(values, [2.5, 97.5], axis=0).T
    return {"top_label_ece_10": intervals[0].tolist(), "ok_class_ece_10": intervals[1].tolist()}


def read_logits(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    files = [r["file"].strip() for r in rows]
    patients = [r["patient_id"].strip() for r in rows]
    if not all(files) or len(set(files)) != len(files) or not all(patients):
        raise ValueError("File IDs must be unique/nonempty and patient IDs nonempty")
    y = np.array([CLASSES.index(r["true_label"]) for r in rows], dtype=int)
    z = np.array([[float(r["logit_" + c]) for c in CLASSES] for r in rows])
    if not np.all(np.isfinite(z)):
        raise ValueError("Nonfinite logits")
    return rows, y, z, np.array(patients)


def check_disjoint(cal_rows, cal_patients, test_rows, test_patients):
    if set(cal_patients) & set(test_patients):
        raise ValueError("Calibration/test patient overlap: evaluation would leak")
    if {r["file"] for r in cal_rows} & {r["file"] for r in test_rows}:
        raise ValueError("Calibration/test file overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--calibration", help="Separate patients; used only to fit temperature")
    parser.add_argument("--out", default="research/calibration_report")
    parser.add_argument("--accept-threshold", type=float, help="Freeze on validation before test; no default")
    args = parser.parse_args()
    if args.accept_threshold is not None and not 0 <= args.accept_threshold <= 1:
        parser.error("accept-threshold must be in [0,1]")
    rows, y, z, patients = read_logits(args.test)
    temperature, calibration_info = 1.0, None
    if args.calibration:
        cal_rows, cy, cz, cp = read_logits(args.calibration)
        check_disjoint(cal_rows, cp, rows, patients)
        temperature = fit_temperature(cy, cz)
        calibration_info = dict(n=len(cy), patients=len(set(cp)),
                                nll_before=nll_from_logits(cy, cz),
                                nll_after=nll_from_logits(cy, cz, temperature))
    variants = {"raw": softmax(z, axis=1)}
    if args.calibration:
        variants["temperature_scaled"] = softmax(z / temperature, axis=1)
    report = dict(class_order=list(CLASSES), temperature=temperature,
                  temperature_at_search_boundary=temperature <= .0501 or temperature >= 19.999,
                  calibration=calibration_info, test_patients=len(set(patients)),
                  bin_convention="[left,right), final bin includes 1; empty bins contribute zero",
                  bootstrap_note="1000 patient cluster resamples; fixed model and fitted temperature; frame-weighted estimand",
                  train_overlap_note="Model-training independence must be checked outside this tool",
                  results={})
    bin_rows = []
    for name, p in variants.items():
        result = evaluate(y, p, args.accept_threshold)
        result["nll_from_logits"] = nll_from_logits(y, z, temperature if name == "temperature_scaled" else 1)
        result["patient_bootstrap_ci95"] = bootstrap_calibration(y, p, patients)
        sources = sorted({r.get("annotator", "") for r in rows} - {""})
        result["by_annotation_source"] = {}
        for source in sources:
            idx = np.array([r.get("annotator") == source for r in rows])
            result["by_annotation_source"][source] = evaluate(y[idx], p[idx], args.accept_threshold)
        report["results"][name] = result
        for kind, conf, outcome in [("top_label", p.max(axis=1), p.argmax(axis=1) == y),
                                    ("ok_class", p[:, 2], y == 2)]:
            _, bins = calibration_bins(conf, outcome)
            bin_rows.extend(dict(variant=name, kind=kind, **b) for b in bins)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "reliability_bins.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, kind in enumerate(("top_label", "ok_class")):
        for name in variants:
            data = [b for b in bin_rows if b["variant"] == name and b["kind"] == kind]
            nonempty = [b for b in data if b["count"]]
            axes[0, column].plot([b["mean_probability"] for b in nonempty],
                                 [b["observed_frequency"] for b in nonempty], "o-", label=name)
            axes[1, column].step([(b["left"] + b["right"]) / 2 for b in data],
                                 [b["count"] for b in data], where="mid", label=name)
        axes[0, column].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[0, column].set(xlim=(0, 1), ylim=(0, 1), title=kind,
                            xlabel="Mean probability", ylabel="Observed frequency")
        axes[1, column].set(xlim=(0, 1), xlabel="Probability bin center", ylabel="Count")
        axes[0, column].legend()
    fig.savefig(out / "reliability.png", dpi=180)
    print(f"Saved {out / 'metrics.json'}; temperature={temperature:.6g}")


if __name__ == "__main__":
    main()

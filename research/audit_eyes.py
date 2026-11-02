"""Read-only review of eyes outputs plus fixed-protocol annotation-source ablation.

Run with the eyes runtime Python. Original models, labels, and outputs are unchanged.
Only this script's sibling eyes_audit_results.json is written.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2] / "eyes")
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from quality_classifier.v2.bootstrap import setup
    setup()
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    output = root / "outputs/quality_classifier_v2"
    rows = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8-sig")))
    records = json.loads((output / "data_snapshot/records.json").read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == [r["file"] for r in records]
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["patient_id"] for r in rows])
    authors = np.array([r["annotator"] for r in rows])
    folds = np.array([int(r["outer_fold"]) for r in rows])
    assert all(len(set(folds[groups == g])) == 1 for g in set(groups))
    X = np.load(output / "cnn_cache/embeddings.npy", mmap_mode="r")[:, :576]
    report = {"scope": "Exploratory diagnostic. Original artifacts read only; no independent validation claim.",
              "sklearn": sklearn.__version__, "class_order": ["not_ok", "ok", "doubtful"],
              "existing_predictions": {}, "source_ablation": {}}
    for family in ["baseline_lr", "svm", "mobilenet_frozen", "mobilenet_finetuned"]:
        p = np.array([[float(r[family + "/p_" + c]) for c in report["class_order"]] for r in rows])
        pred = p.argmax(axis=1)
        clear = y != 2
        thresholds = []
        for t in [.5, .6, .7, .8, .9, .95]:
            keep = p[:, 1] >= t
            thresholds.append({"threshold": t, "accepted": int(keep.sum()),
                               "accepted_counts": np.bincount(y[keep], minlength=3).tolist(),
                               "ok_precision": float((y[keep] == 1).mean()) if keep.any() else None,
                               "ok_recall": float(keep[y == 1].mean()), "coverage": float(keep.mean())})
        report["existing_predictions"][family] = {
            "accuracy": float(accuracy_score(y, pred)),
            "matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "total_errors": int((pred != y).sum()),
            "errors_touching_doubtful": int(((pred != y) & ((pred == 2) | (y == 2))).sum()),
            "auc_ok_vs_rest": float(roc_auc_score(y == 1, p[:, 1])),
            "clear_only_binary_auc": float(roc_auc_score(y[clear] == 1, p[clear, 1] / (p[clear, 0] + p[clear, 1]))),
            "clear_only_binary_accuracy": float(np.mean((p[clear, 1] > p[clear, 0]) == (y[clear] == 1))),
            "illustrative_thresholds_not_chosen_on_validation": thresholds,
        }
    configs = [("all_balanced", "all", "balanced"), ("human_balanced", "human", "balanced"),
               ("assistant_balanced", "assistant", "balanced"), ("all_unweighted", "all", None)]
    with threadpool_limits(limits=1):
        for name, train_author, weight in configs:
            predictions = np.full(len(y), -1, dtype=int)
            training_sizes = []
            for outer in range(5):
                train = (folds != outer) & (folds != ((outer + 1) % 5))
                if train_author != "all":
                    train &= authors == train_author
                test = folds == outer
                assert not set(groups[train]) & set(groups[test])
                assert set(y[train]) == {0, 1, 2}
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.01, class_weight=weight, max_iter=2500, random_state=42 + outer))
                model.fit(X[train], y[train])
                predictions[test] = model.predict(X[test])
                training_sizes.append(int(train.sum()))
            result = {"training_sizes": training_sizes, "by_test_source": {}}
            for source in ["all", "human", "assistant"]:
                mask = np.ones(len(y), dtype=bool) if source == "all" else authors == source
                matrix = confusion_matrix(y[mask], predictions[mask], labels=[0, 1, 2])
                result["by_test_source"][source] = {
                    "n": int(mask.sum()), "accuracy": float(accuracy_score(y[mask], predictions[mask])),
                    "macro_f1": float(f1_score(y[mask], predictions[mask], average="macro", labels=[0, 1, 2], zero_division=0)),
                    "matrix": matrix.tolist(),
                    "ok_precision": float(matrix[1, 1] / matrix[:, 1].sum()) if matrix[:, 1].sum() else None,
                    "ok_recall": float(matrix[1, 1] / matrix[1].sum()),
                }
            # Paired patient bootstrap vs mixed-source balanced head on human test labels.
            if name == "all_balanced":
                reference = predictions.copy()
            else:
                idx_by_patient = [np.flatnonzero((groups == g) & (authors == "human")) for g in np.unique(groups)]
                rng = np.random.default_rng(20261101)
                accuracy_delta = []
                for _ in range(1000):
                    idx = np.concatenate([idx_by_patient[g] for g in rng.integers(len(idx_by_patient), size=len(idx_by_patient))])
                    accuracy_delta.append(np.mean(predictions[idx] == y[idx]) - np.mean(reference[idx] == y[idx]))
                result["human_accuracy_delta_vs_all_balanced_ci95"] = np.percentile(accuracy_delta, [2.5, 97.5]).tolist()
            report["source_ablation"][name] = result
            print(name, json.dumps(result), flush=True)
    report["ablation_protocol"] = {
        "features": "Frozen ImageNet full-frame 320x240, 576 features; no center crop",
        "folds": "Existing five patient folds; test=outer, excluded calibration=(outer+1)%5, training=remaining three",
        "model": "StandardScaler + multinomial LogisticRegression C=.01; fixed settings, no search",
        "calibration": "Not fitted: argmax diagnostic, not probability or threshold validation",
        "limitation": "Source-specific models also use fewer training examples; source and sample size effects are not fully isolated",
    }
    report["current_annotation_hashes"] = {
        n: hashlib.sha256((root / "picture-quality-analysis" / n).read_bytes()).hexdigest()
        for n in ["frames.json", "frames_assistant.json"]}
    path = Path(__file__).with_name("eyes_audit_results.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved", str(path).encode("ascii", errors="backslashreplace").decode())


if __name__ == "__main__":
    main()

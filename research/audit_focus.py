"""Reproducible dataset audit and exploratory focus-score analysis, not a model test."""
import csv
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "capillary_quality_1000"
OUT = Path(__file__).resolve().parent
SCORES = ["laplacian_raw", "laplacian_smooth", "tenengrad_smooth", "local_tenengrad_p75"]


def main():
    cv2.setNumThreads(1)
    frames = json.loads((DATA / "frames.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    with (DATA / "patient_map.csv").open(encoding="utf-8-sig", newline="") as handle:
        mappings = list(csv.DictReader(handle))
    patients = {x["video"]: x["patient_id"] for x in mappings}
    expected = {x["file"]: x for x in manifest["images_sha256"]}
    rows, sizes, modes, hashes = [], Counter(), Counter(), defaultdict(list)
    errors, timings = [], []
    for i, frame in enumerate(frames):
        file = frame["file"]
        path = DATA / "pictures" / file
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        hashes[digest].append(file)
        if digest != expected[file]["sha256"] or len(raw) != expected[file]["bytes"]:
            errors.append(file)
        with Image.open(path) as im:
            sizes[str(im.size)] += 1
            modes[im.mode] += 1
        start = time.perf_counter()
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError(f"Cannot decode {file}")
        decode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        # Fixed pixel scale; no per-image contrast normalization or content masking.
        gray = decoded.astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
        lap_raw = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())
        lap_smooth = float(cv2.Laplacian(smooth, cv2.CV_32F, ksize=1).var())
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        energy = gx * gx + gy * gy
        tiles = [float(tile.mean()) for band in np.array_split(energy, 4)
                 for tile in np.array_split(band, 4, axis=1)]
        score_seconds = time.perf_counter() - start
        timings.append((decode_seconds, score_seconds))
        label = "ok" if frame["quality_ok"] is True else "not_ok" if frame["quality_ok"] is False else "doubtful"
        video = Path(file).stem.rsplit("_", 1)[0] + ".avi"
        rows.append(dict(file=file, video=video, patient_id=patients[video],
                         label=label, annotator=frame["annotator"],
                         laplacian_raw=lap_raw, laplacian_smooth=lap_smooth,
                         tenengrad_smooth=float(energy.mean()),
                         local_tenengrad_p75=float(np.percentile(tiles, 75))))
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(frames)}", flush=True)
    with (OUT / "focus_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Binary clear-label ranking is exploratory. Doubtful is retained in CSV/plots.
    clear = [r for r in rows if r["label"] != "doubtful"]
    labels = np.array([r["label"] == "ok" for r in clear])
    groups = np.array([r["patient_id"] for r in clear])
    unique_groups = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(20261101)
    bootstrap = []
    for _ in range(1000):
        idx = np.concatenate([indices[g] for g in rng.choice(unique_groups, len(unique_groups), replace=True)])
        if len(np.unique(labels[idx])) == 2:
            bootstrap.append(idx)
    results = {}
    for score in SCORES:
        values = np.array([r[score] for r in clear])
        distribution = [roc_auc_score(labels[idx], values[idx]) for idx in bootstrap]
        by_annotator = {}
        for annotator in ("human", "assistant"):
            keep = np.array([r["annotator"] == annotator for r in clear])
            by_annotator[annotator] = float(roc_auc_score(labels[keep], values[keep]))
        results[score] = {
            "auc_ok_vs_not_ok": float(roc_auc_score(labels, values)),
            "patient_bootstrap_percentile_ci95": np.percentile(distribution, [2.5, 97.5]).tolist(),
            "auc_by_annotation_source": by_annotator,
            "median_by_class": {c: float(np.median([r[score] for r in rows if r["label"] == c]))
                                for c in ("not_ok", "doubtful", "ok")},
        }
    per_patient = Counter(r["patient_id"] for r in rows)
    per_video = Counter(r["video"] for r in rows)
    report = {
        "scope": "Exploratory ranking on all available data; no trained classifier, independent test, or calibration claim.",
        "images": len(rows), "image_files": len(list((DATA / "pictures").glob("*.png"))),
        "videos": len(per_video), "patients": len(per_patient),
        "classes": dict(Counter(r["label"] for r in rows)),
        "annotation_sources": dict(Counter(r["annotator"] for r in rows)),
        "classes_by_annotation_source": {a: dict(Counter(r["label"] for r in rows if r["annotator"] == a))
                                         for a in ("human", "assistant")},
        "image_sizes": dict(sizes), "image_modes": dict(modes),
        "frames_per_video_histogram": dict(Counter(per_video.values())),
        "frames_per_patient_min_median_max": [min(per_patient.values()), float(np.median(list(per_patient.values()))), max(per_patient.values())],
        "duplicate_annotation_names": len(frames) - len({r["file"] for r in rows}),
        "duplicate_video_map_rows": len(mappings) - len(patients),
        "duplicate_image_hash_groups": [names for names in hashes.values() if len(names) > 1],
        "manifest_mismatches": errors,
        "annotation_sha256_matches_manifest": hashlib.sha256((DATA / "frames.json").read_bytes()).hexdigest() == manifest["annotation_sha256"],
        "unannotated_image_files": sorted({p.name for p in (DATA / "pictures").glob("*.png")} - {r["file"] for r in rows}),
        "clear_label_ranking_n": len(clear), "bootstrap_repetitions": len(bootstrap),
        "scores": results,
        "median_decode_ms": float(np.median(np.array(timings)[:, 0]) * 1000),
        "median_all_four_scores_ms_excluding_decode_io_hash": float(np.median(np.array(timings)[:, 1]) * 1000),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__, "opencv_threads": 1},
    }
    (OUT / "audit_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ["Raw Laplacian", "Smoothed Laplacian", "Smoothed Tenengrad", "Local Tenengrad p75"]
    aucs = np.array([results[s]["auc_ok_vs_not_ok"] for s in SCORES])
    bounds = np.array([results[s]["patient_bootstrap_percentile_ci95"] for s in SCORES])
    axes[0].errorbar(aucs, range(4), xerr=[aucs - bounds[:, 0], bounds[:, 1] - aucs], fmt="o", capsize=4)
    axes[0].set(yticks=range(4), yticklabels=names, xlim=(0, 1), xlabel="AUROC: OK vs not OK", title="Exploratory scores; patient bootstrap 95% CI")
    axes[0].axvline(.5, color="gray", linestyle="--")
    for c, color in [("not_ok", "#cb4b4b"), ("doubtful", "#c49a26"), ("ok", "#278c69")]:
        vals = [np.log10(max(r["tenengrad_smooth"], 1e-12)) for r in rows if r["label"] == c]
        axes[1].hist(vals, bins=25, density=True, histtype="step", linewidth=2, label=c, color=color)
    axes[1].set(xlabel="log10(smoothed Tenengrad)", ylabel="Density", title="All 3 classes; overlapping score distributions")
    axes[1].legend()
    fig.savefig(OUT / "focus_scores.png", dpi=180)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

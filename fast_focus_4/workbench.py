"""Standalone notebook utilities; all models and tabular data live in this folder."""
import hashlib
import io
import json
import platform
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from PIL import Image

from .core import BASIC, VERSION, from_gray, fit_bundle, group_splits, predict_bundle
from .core import decision_metrics, probability_metrics

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
PICTURES = Path(os.environ.get('CAPILLARY_PICTURES', HERE / 'pictures')).expanduser()
SEED = 20261102
MODEL_IDS = ('logistic_basic', 'trees_basic')


def load_data():
    """Portable snapshot with four features, labels, groups, hashes, and OOF scores."""
    table = pd.read_csv(ART / 'dataset.csv', dtype={'file': str, 'patient_id': str}).fillna({'review_comment': ''})
    assert table.file.is_unique and len(table) == 1000
    assert (table.y == 1).sum() == 62 and (table.y == 0).sum() == 503
    assert set(table.y) == {-1, 0, 1}
    assert np.isfinite(table[BASIC].to_numpy(float)).all()
    return table


def verify_artifacts():
    manifest = json.loads((ART / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifacts'].items():
        path = ART / name
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return manifest


def baseline_table():
    report = json.loads((ART / 'metrics.json').read_text(encoding='utf-8'))
    speed = json.loads((ART / 'speed.json').read_text(encoding='utf-8'))
    rows = []
    for model in MODEL_IDS:
        m = report['models'][model]
        rows.append(dict(model=model, auroc=m['calibrated']['auroc'],
                         auroc_ci95=str(m['ci95']['auroc']),
                         ap=m['calibrated']['average_precision'],
                         ece_ok_10_raw=m['raw']['ece_ok']['10'],
                         ece_ok_10=m['calibrated']['ece_ok']['10'],
                         precision_ok=m['balanced']['precision_ok'],
                         recall_ok=m['balanced']['recall_ok'],
                         tp=m['balanced']['tp'], fp=m['balanced']['fp'],
                         fn=m['balanced']['fn'], tn=m['balanced']['tn'],
                         png_to_decision_p50_ms=speed['models'][model]['end_to_end_ms_p50']))
    return pd.DataFrame(rows).set_index('model')


def load_model(model='logistic_basic'):
    if model not in MODEL_IDS:
        raise ValueError(f'Choose one of {MODEL_IDS}')
    bundle = joblib.load(ART / 'models.joblib')[model]
    assert bundle['features'] == BASIC
    return bundle


def decision_text(p, threshold):
    if threshold > 1:
        return 'auto_accept_disabled'
    return 'ok' if p >= threshold else 'not_ok'


def predict_files(files, model='logistic_basic', mode='balanced', threshold=None):
    """Native 1024x768 only. No silent resize; known training images are marked."""
    if mode not in ('balanced', 'precise'):
        raise ValueError('mode must be balanced or precise')
    if threshold is not None and not (np.isfinite(threshold) and 0 <= threshold <= 1):
        raise ValueError('Manual threshold must lie in [0, 1]')
    b = load_model(model)
    t = b['thresholds'][mode] if threshold is None else float(threshold)
    cached = load_data()
    known = set(cached.loc[cached.y >= 0, 'sha256'])
    dataset_hashes = set(cached.sha256)
    cv2.setNumThreads(1)
    result = []
    for file in files:
        path = Path(file)
        start = time.perf_counter()
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode not in ('L', 'RGB', 'RGBA') or getattr(image, 'n_frames', 1) != 1:
                raise ValueError(f'Expected a single 8-bit grayscale/RGB/RGBA frame: {path}')
        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (768, 1024):
            raise ValueError(f'Expected native 1024x768 image: {path}')
        f = from_gray(gray, basic_only=True)
        p, _ = predict_bundle(b, np.array([[f[key] for key in BASIC]]))
        elapsed = (time.perf_counter() - start) * 1000
        digest = hashlib.sha256(raw).hexdigest()
        result.append(dict(file=str(path), model=model, p_ok=float(p[0]), threshold=t,
                           decision=decision_text(float(p[0]), t),
                           known_training_image=digest in known, known_dataset_image=digest in dataset_hashes,
                           read_decode_features_predict_ms=elapsed, **f))
    return pd.DataFrame(result)


def benchmark(files, model='logistic_basic', warmup=3):
    """Read/decode, features, head separately. Exclude loading and warmup."""
    files = [Path(f) for f in files]
    if not files:
        raise ValueError('No benchmark images')
    if warmup < 0:
        raise ValueError('warmup must be nonnegative')
    cv2.setNumThreads(1)
    bundle = load_model(model)
    measured = []
    for i, file in enumerate([files[j % len(files)] for j in range(warmup)] + files):
        start = time.perf_counter()
        a = cv2.imdecode(np.frombuffer(file.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None or a.shape != (768, 1024):
            raise ValueError(f'Expected native 1024x768 image: {file}')
        decoded = time.perf_counter()
        f = from_gray(a, basic_only=True)
        featured = time.perf_counter()
        predict_bundle(bundle, np.array([[f[key] for key in BASIC]]))
        end = time.perf_counter()
        if i >= warmup:
            measured.append(dict(file=file.name, read_decode_ms=(decoded-start)*1000,
                                 features_ms=(featured-decoded)*1000,
                                 head_ms=(end-featured)*1000, total_ms=(end-start)*1000))
    return pd.DataFrame(measured)


def experiment_specs():
    """Fixed ablations: keep model, resolution, calibration, and splits unchanged."""
    return [('baseline_4', list(BASIC))] + [
        ('without_' + feature, [f for f in BASIC if f != feature]) for feature in BASIC
    ]


def run_ablations(table, output_dir=None):
    """Train LR-4 and four leave-one-feature-out variants; outer labels never fit.

    Exploratory comparison, not hyperparameter search or automatic deployment.
    No write to the production artifacts directory.
    """
    labeled = table.loc[table['y'] >= 0].reset_index(drop=True)
    y = labeled['y'].to_numpy(dtype=int)
    groups = labeled['patient_id'].to_numpy()
    splits = group_splits(y, groups, n=5, seed=SEED)
    metrics, predictions, fold_metrics = [], [], []
    for name, features in experiment_specs():
        x = labeled[features].to_numpy(dtype=float)
        prob = np.full(len(y), np.nan)
        raw, balanced, precise = [prob.copy() for _ in range(3)]
        start = time.perf_counter()
        for fold, (train, valid) in enumerate(splits):
            assert not set(groups[train]) & set(groups[valid])
            assert np.all(labeled.loc[valid, 'fold'].to_numpy() == fold)
            b = fit_bundle(x[train], y[train], groups[train], 'logistic', SEED + fold * 10)
            prob[valid], raw[valid] = predict_bundle(b, x[valid])
            balanced[valid] = b['thresholds']['balanced']
            precise[valid] = b['thresholds']['precise']
            fold_metrics.append(dict(experiment=name, fold=fold,
                                     auroc=probability_metrics(y[valid], prob[valid])['auroc'],
                                     **decision_metrics(y[valid], prob[valid], balanced[valid])))
        assert np.isfinite(prob).all()
        # Detect accidental drift before interpreting an improvement.
        if name == 'baseline_4':
            np.testing.assert_allclose(prob, labeled['logistic_basic__p'], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(balanced, labeled['logistic_basic__balanced'], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(precise, labeled['logistic_basic__precise'], rtol=1e-7, atol=1e-9)
        pm = probability_metrics(y, prob)
        dm = decision_metrics(y, prob, balanced)
        metrics.append(dict(experiment=name, n_features=len(features), features=';'.join(features),
                            auroc=pm['auroc'], ap=pm['average_precision'],
                            ece_ok_10=pm['ece_ok']['10'], brier=pm['brier'],
                            **dm, strict_accepted=int((prob >= precise).sum()),
                            cv_training_seconds=time.perf_counter()-start))
        for i, row in labeled.iterrows():
            predictions.append(dict(experiment=name, file=row['file'], patient_id=row['patient_id'],
                                    fold=int(row['fold']), y=int(y[i]), p_ok=prob[i], p_raw=raw[i],
                                    threshold_balanced=balanced[i], threshold_precise=precise[i]))
        print(f'{name}: AUROC={pm["auroc"]:.4f}, AP={pm["average_precision"]:.4f}', flush=True)
    result = pd.DataFrame(metrics)
    result['auroc_delta_vs_baseline'] = result['auroc'] - result.iloc[0]['auroc']
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.to_csv(out/'ablation_metrics.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(predictions).to_csv(out/'ablation_oof.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(fold_metrics).to_csv(out/'ablation_folds.csv', index=False, encoding='utf-8-sig')
        sources = [ART/'dataset.csv', ART/'models.joblib']
        metadata = dict(seed=SEED, python=platform.python_version(), sklearn=sklearn.__version__,
                        numpy=np.__version__, outer_folds=5, inner_folds=3,
                        feature_version=VERSION,
                        sources_sha256={p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sources},
                        note='Exploratory patient-grouped ablations. No final model replaced; no external test.')
        (out/'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return result, pd.DataFrame(predictions), pd.DataFrame(fold_metrics)


def photo_grid(table, model='logistic_basic', per_case=2):
    """Most confident false accepts/rejects, plus correct examples; all scores OOF."""
    import matplotlib.pyplot as plt
    from PIL import Image
    d = table.loc[table['y'] >= 0].copy()
    d['p'] = d[f'{model}__p']
    d['accepted'] = d['p'] >= d[f'{model}__balanced']
    cases = [('TP: good accepted', (d.y == 1) & d.accepted, False),
             ('FP: bad accepted', (d.y == 0) & d.accepted, False),
             ('FN: good rejected', (d.y == 1) & ~d.accepted, True),
             ('TN: bad rejected', (d.y == 0) & ~d.accepted, True)]
    fig, axes = plt.subplots(4, per_case, figsize=(6*per_case, 13), squeeze=False, layout='constrained')
    for row_idx, (title, mask, ascending) in enumerate(cases):
        sample = d.loc[mask].sort_values('p', ascending=ascending).head(per_case)
        for ax in axes[row_idx]:
            ax.axis('off')
        for ax, (_, row) in zip(axes[row_idx], sample.iterrows()):
            path = PICTURES/row['file']
            if path.is_file():
                with Image.open(path) as im:
                    ax.imshow(im)
            else:
                ax.text(.5, .5, 'Original image not included. Set CAPILLARY_PICTURES.', ha='center', wrap=True)
            ax.set_title(f'{title}; OOF P(ok)={row.p:.3f}\n{row["file"]}', fontsize=9)
    return fig

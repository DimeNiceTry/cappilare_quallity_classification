"""Standalone four-feature implementation, frozen from native-quality-v1.
Training, calibration, and metrics preserve the original experiment.
"""
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score
"""Fixed models and calibration fitted without external evaluation patients."""
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold


def log_features(x):
    return np.log10(np.maximum(np.asarray(x, float), 1e-10))


def estimator(kind, seed):
    if kind == 'trees':
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, max_depth=10,
                                    class_weight='balanced', max_features=.8, n_jobs=2, random_state=seed)
    transform = FunctionTransformer(log_features) if kind != 'neural' else FunctionTransformer()
    return make_pipeline(transform, StandardScaler(),
                         LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=seed))


def group_splits(y, groups, n=3, seed=20261102):
    splits = list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros((len(y),1)), y, groups))
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
        assert len(np.unique(y[train])) == len(np.unique(y[test])) == 2
    return splits


def choose_thresholds(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    thresholds = np.r_[0., np.unique(p), np.nextafter(float(p.max()), np.inf)]
    best, conservative = None, None
    pos, neg = int(y.sum()), int((1-y).sum())
    for t in thresholds:
        accepted = p >= t
        tp, fp = int(y[accepted].sum()), int((1-y[accepted]).sum())
        recall, specificity = tp/pos, 1-fp/neg
        key = ((recall+specificity)/2, -fp, float(t))
        if best is None or key > best[0]: best=(key,float(t))
        if tp+fp >= 5 and tp/(tp+fp) >= .90:
            key=(tp, -fp, float(t))
            if conservative is None or key>conservative[0]: conservative=(key,float(t))
    return dict(balanced=best[1], precise=conservative[1] if conservative else 2.0)


def fit_bundle(x, y, groups, kind, seed):
    """Inner OOF scores fit sigmoid and thresholds; external labels never enter."""
    if kind == 'constant':
        return dict(kind=kind, prevalence=float(np.mean(y)), thresholds={'balanced':2., 'precise':2.})
    oof = np.full(len(y), np.nan)
    splits = group_splits(y, groups, seed=seed)
    for i, (train, valid) in enumerate(splits):
        m=estimator(kind,seed+i);m.fit(x[train],y[train])
        oof[valid]=m.predict_proba(x[valid])[:,1]
    assert np.isfinite(oof).all()
    calibrator=LogisticRegression(C=100., max_iter=1000)
    calibrator.fit(logit(np.clip(oof,1e-6,1-1e-6)).reshape(-1,1),y)
    calibrated=calibrator.predict_proba(logit(np.clip(oof,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    model=estimator(kind,seed);model.fit(x,y)
    return dict(kind=kind,model=model,calibrator=calibrator,thresholds=choose_thresholds(y,calibrated),
                calibration_training_n=len(y), inner_patient_splits=[
                    dict(train=sorted(set(groups[t])),valid=sorted(set(groups[v]))) for t,v in splits])


def predict_bundle(bundle,x):
    if bundle['kind']=='constant':
        p=np.full(len(x),bundle['prevalence']);return p,p.copy()
    if bundle['kind']=='trees':
        bundle['model'].n_jobs=1  # Avoid thread-pool overhead for one photograph.
    raw=bundle['model'].predict_proba(x)[:,1]
    p=bundle['calibrator'].predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    return p,raw

VERSION = 'native-quality-v1'
BASIC = ['laplacian_raw', 'laplacian_smooth', 'tenengrad_smooth', 'local_tenengrad_p75']

def tiles(a):
    return [t for band in np.array_split(a, 4) for t in np.array_split(band, 4, axis=1)]

def gradient(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, gx*gx + gy*gy

def from_gray(gray, basic_only=False):
    a = gray.astype(np.float32) / 255.0
    s = cv2.GaussianBlur(a, (0, 0), 1.0)
    gx, gy, energy = gradient(s)
    local = np.array([float(t.mean()) for t in tiles(energy)])
    f = dict(laplacian_raw=float(cv2.Laplacian(a, cv2.CV_32F, ksize=1).var()),
             laplacian_smooth=float(cv2.Laplacian(s, cv2.CV_32F, ksize=1).var()),
             tenengrad_smooth=float(energy.mean()), local_tenengrad_p75=float(np.percentile(local, 75)))
    return f

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

def decision_metrics(y,p,threshold):
    accepted=p>=threshold;tp=int(np.sum(accepted & (y==1)));fp=int(np.sum(accepted & (y==0)))
    fn=int(np.sum(~accepted & (y==1)));tn=int(np.sum(~accepted & (y==0)))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,accepted=tp+fp,accuracy=(tp+tn)/len(y),
                balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                precision_ok=tp/(tp+fp) if tp+fp else None,recall_ok=tp/(tp+fn),
                false_accept_rate=fp/(fp+tn),coverage=float(accepted.mean()))

def probability_metrics(y,p):
    two=np.column_stack([1-p,p])
    return dict(auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(np.clip(p,1e-15,1))+(1-y)*np.log(np.clip(1-p,1e-15,1)))),
                ece_ok={str(n):calibration_bins(p,y,n)[0] for n in (5,10,15)},
                ece_top={str(n):top_label_ece(y,two,n) for n in (5,10,15)})

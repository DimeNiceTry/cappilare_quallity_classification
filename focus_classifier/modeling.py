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

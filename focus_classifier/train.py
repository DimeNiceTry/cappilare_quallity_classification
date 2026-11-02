"""Train fixed feature models; produce patient-held-out scores and final models."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import sklearn
from .features import BASIC
from .modeling import group_splits, fit_bundle, predict_bundle

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'focus_classifier/artifacts'


def dataset():
    export=ROOT/'data/annotations'
    if not (export/'all_comparisons.csv').is_file():
        latest=json.loads((ROOT/'review_client/data/latest_export.json').read_text(encoding='utf-8'))
        export=Path(latest['path'])
        if not export.is_absolute():export=ROOT/export
    rows=list(csv.DictReader((export/'all_comparisons.csv').open(encoding='utf-8-sig',newline='')))
    y=np.array([int(r['human_label']=='ok') if r['status']=='agreed_binary' else -1 for r in rows])
    groups=np.array([r['patient_id'] for r in rows])
    assert (y==1).sum()==62 and (y==0).sum()==503
    return rows,y,groups,export


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--neural-only',action='store_true');args=parser.parse_args()
    rows,y,groups,export=dataset()
    labels=np.flatnonzero(y>=0)
    if args.neural_only:
        emb=np.load(ART/'neural_embeddings.npz')
        lookup={f:i for i,f in enumerate(emb['files'])}
        x=np.array([emb['features'][lookup[r['file']]] for r in rows])
        names=[f'embedding_{i}' for i in range(x.shape[1])]
        specs=[dict(id='neural',name='Нейросеть MobileNetV3 + линейная голова',kind='neural',features=names)]
    else:
        features=list(csv.DictReader((ART/'features.csv').open(encoding='utf-8')))
        lookup={r['file']:r for r in features};names=[n for n in features[0] if n not in ('file','sha256')]
        x=np.array([[float(lookup[r['file']][n]) for n in names] for r in rows])
        specs=[dict(id='constant',name='Всегда не годен',kind='constant',features=BASIC)]
        titles=['Лапласиан исходный','Лапласиан после сглаживания','Tenengrad всего кадра','Локальный Tenengrad p75']
        specs += [dict(id='single_'+f,name=t,kind='logistic',features=[f]) for f,t in zip(BASIC,titles)]
        specs += [dict(id=k+'_'+v,name=t+' · '+('4 признака' if v=='basic' else f'{len(names)} признаков'),kind=k,features=fs)
                  for v,fs in [('basic',BASIC),('extended',names)] for k,t in [('logistic','Логистическая регрессия'),('trees','ExtraTrees')]]
    assert np.isfinite(x).all()
    splits=group_splits(y[labels],groups[labels],n=5)
    fold=np.full(len(rows),-1,int);manifest=[]
    for i,(tr,te) in enumerate(splits):
        train,test=labels[tr],labels[te];test_patients=set(groups[test])
        fold[np.isin(groups,list(test_patients))]=i
        manifest.append(dict(fold=i,train_patients=sorted(set(groups[train])),test_patients=sorted(test_patients),
                             train_files=[rows[j]['file'] for j in train],test_files=[rows[j]['file'] for j in test],
                             train_n=len(train),train_ok=int(y[train].sum()),test_n=len(test),test_ok=int(y[test].sum())))
    assert (fold>=0).all(), 'Patient without consensus label needs explicit fold assignment'
    (ART/'folds.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    output={'fold':fold,'y':y,'files':np.array([r['file'] for r in rows]),'patients':groups}
    final={};thresholds={}
    for spec in specs:
        name=spec['id'];xx=x[:,[names.index(f) for f in spec['features']]]
        p=np.full(len(rows),np.nan);raw=p.copy();balanced=p.copy();precise=p.copy()
        thresholds[name]=[]
        for i, (tr,te) in enumerate(splits):
            train=labels[tr];test_all=np.flatnonzero(fold==i)
            assert not set(groups[train]) & set(groups[test_all])
            b=fit_bundle(xx[train],y[train],groups[train],spec['kind'],20261102+i*10)
            p[test_all],raw[test_all]=predict_bundle(b,xx[test_all])
            balanced[test_all]=b['thresholds']['balanced'];precise[test_all]=b['thresholds']['precise']
            thresholds[name].append(dict(fold=i,**b['thresholds'],calibration_patients=sorted(set(groups[train]))))
            print(f'{name}: outer fold {i+1}/5',flush=True)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        for k,v in [('p',p),('raw',raw),('balanced',balanced),('precise',precise)]:output[name+'__'+k]=v
        b=fit_bundle(xx[labels],y[labels],groups[labels],spec['kind'],20261102)
        final[name]=dict(**b,features=spec['features'],name=spec['name'])
        print(f'{name}: final model saved in memory',flush=True)
    suffix='neural' if args.neural_only else 'features'
    np.savez_compressed(ART/f'predictions_{suffix}.npz',**output)
    joblib.dump(final,ART/f'models_{suffix}.joblib',compress=3)
    metadata=dict(specs=specs,thresholds=thresholds,source_export=export.relative_to(ROOT).as_posix(),
                  annotation_hash=hashlib.sha256((export/'frames.json').read_bytes()).hexdigest(),
                  seed=20261102,python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__)
    (ART/f'models_{suffix}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Completed {suffix}',flush=True)


if __name__=='__main__':main()

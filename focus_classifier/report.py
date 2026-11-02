"""OOF metrics, clustered intervals, plots, and gallery data."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from research.evaluate_calibration import calibration_bins, top_label_ece
from .features import BASIC
from .train import ART, ROOT, dataset


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


def bootstrap_indices(groups):
    unique=np.unique(groups);ids={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(20261102)
    return [np.concatenate([ids[g] for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(1000)]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    rows,y,groups,export=dataset();keep=y>=0;yy=y[keep];boot=bootstrap_indices(groups[keep])
    arrays={};specs=[]
    for kind in ('features','neural'):
        path=ART/f'predictions_{kind}.npz'
        if not path.exists():continue
        a=np.load(path)
        assert np.array_equal(a['y'],y) and np.array_equal(a['files'],[r['file'] for r in rows])
        arrays.update({k:a[k] for k in a.files})
        specs += json.loads((ART/f'models_{kind}.json').read_text(encoding='utf-8'))['specs']
    result={};samples={};bins=[];table=[]
    for spec in specs:
        name=spec['id'];p=arrays[name+'__p'][keep];raw=arrays[name+'__raw'][keep];t=arrays[name+'__balanced'][keep]
        item=dict(name=spec['name'],features=len(spec['features']),calibrated=probability_metrics(yy,p),raw=probability_metrics(yy,raw),
                  balanced=decision_metrics(yy,p,t),precise=decision_metrics(yy,p,arrays[name+'__precise'][keep]),
                  at_half=decision_metrics(yy,p,.5),folds=[])
        for fold in range(5):
            k=arrays['fold'][keep]==fold
            item['folds'].append(dict(fold=fold,n=int(k.sum()),ok=int(yy[k].sum()),auroc=float(roc_auc_score(yy[k],p[k])),
                                      **decision_metrics(yy[k],p[k],t[k])))
        samples[name]=[]
        for idx in boot:
            if len(np.unique(yy[idx]))<2:continue
            d=decision_metrics(yy[idx],p[idx],t[idx])
            samples[name].append([roc_auc_score(yy[idx],p[idx]),average_precision_score(yy[idx],p[idx]),
                                  d['balanced_accuracy'],d['precision_ok'] if d['precision_ok'] is not None else np.nan,d['recall_ok'],
                                  calibration_bins(p[idx],yy[idx],10)[0]])
        sample_array=np.asarray(samples[name])
        ci=[np.percentile(col[np.isfinite(col)],[2.5,97.5]).tolist() if np.isfinite(col).any() else [None,None] for col in sample_array.T]
        item['ci95']=dict(zip(['auroc','average_precision','balanced_accuracy','precision_ok','recall_ok','ece_ok_10'],ci))
        for mode,threshold in [('precise',arrays[name+'__precise'][keep]),('at_half',.5)]:
            values=[]
            for idx in boot:
                if len(np.unique(yy[idx]))<2:continue
                th=threshold[idx] if isinstance(threshold,np.ndarray) else threshold
                d=decision_metrics(yy[idx],p[idx],th)
                values.append([d['precision_ok'] if d['precision_ok'] is not None else np.nan,d['recall_ok'],d['coverage']])
            bounds=[np.percentile(col[np.isfinite(col)],[2.5,97.5]).tolist() if np.isfinite(col).any() else [None,None] for col in np.array(values).T]
            item[mode]['ci95']=dict(zip(['precision_ok','recall_ok','coverage'],bounds))
        result[name]=item
        for variant,prob in [('raw',raw),('calibrated',p)]:
            for kind,conf,outcome in [('ok',prob,yy),('top',np.maximum(prob,1-prob),(prob>.5)==yy)]:
                _,bs=calibration_bins(conf,outcome,10)
                bins.extend(dict(model=name,variant=variant,kind=kind,**b) for b in bs)
        table.append(dict(model=name,name=spec['name'],auroc=item['calibrated']['auroc'],ap=item['calibrated']['average_precision'],
                          **item['balanced'],ece_ok_10=item['calibrated']['ece_ok']['10'],ece_top_10=item['calibrated']['ece_top']['10'],
                          brier=item['calibrated']['brier'],log_loss=item['calibrated']['log_loss']))
        print(f'Metrics: {name}',flush=True)
    comparisons={}
    for a,b in [('trees_extended','trees_basic'),('logistic_extended','logistic_basic'),('neural','trees_extended'),('neural','logistic_extended')]:
        if a in samples and b in samples:
            delta=np.asarray(samples[a])-np.asarray(samples[b])
            comparisons[a+'_minus_'+b]={metric:np.nanpercentile(delta[:,i],[2.5,97.5]).tolist() for i,metric in enumerate(['auroc','ap','balanced_accuracy','precision','recall','ece'])}
    fs=list(csv.DictReader((ART/'features.csv').open(encoding='utf-8')));fl={r['file']:r for r in fs}
    raw_features={}
    for feature in BASIC:
        values=np.array([float(fl[r['file']][feature]) for r in rows])[keep]
        distr=[roc_auc_score(yy[idx],values[idx]) for idx in boot if len(np.unique(yy[idx]))==2]
        raw_features[feature]=dict(auroc=float(roc_auc_score(yy,values)),ci95=np.percentile(distr,[2.5,97.5]).tolist())
    gallery=[];flat=[]
    for i,r in enumerate(rows):
        scores={}
        for spec in specs:
            name=spec['id'];p=float(arrays[name+'__p'][i]);t=float(arrays[name+'__balanced'][i]);tp=float(arrays[name+'__precise'][i])
            scores[name]=dict(p=p,raw=float(arrays[name+'__raw'][i]),threshold=t,precise_threshold=tp)
            flat.append(dict(file=r['file'],patient_id=r['patient_id'],fold=int(arrays['fold'][i]),status=r['status'],
                             consensus_label=r['human_label'] if y[i]>=0 else '',human_label=r['human_label'],assistant_label=r['assistant_label'],
                             model=name,p_ok=p,p_ok_raw=float(arrays[name+'__raw'][i]),threshold_balanced=t,
                             predicted_balanced='ok' if p>=t else 'not_ok',threshold_precise=tp,predicted_precise='ok' if p>=tp else 'not_ok'))
        gallery.append(dict(file=r['file'],patient=r['patient_id'],fold=int(arrays['fold'][i]),status=r['status'],
                            truth=int(y[i]),human=r['human_label'],assistant=r['assistant_label'],comment=r['review_comment'],
                            features={k:float(fl[r['file']][k]) for k in BASIC},scores=scores))
    report=dict(n=565,ok=62,not_ok=503,patients=len(set(groups[keep])),models=result,raw_feature_auc=raw_features,
                paired_delta_ci95=comparisons,bootstrap='1000 patient cluster resamples, fixed OOF predictions; not retraining intervals',
                evaluation='5 outer patient folds; 3 inner patient folds for sigmoid calibration and thresholds; exploratory, not final external test')
    (ART/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (ART/'gallery.json').write_text(json.dumps(dict(models=[dict(id=s['id'],name=s['name']) for s in specs],rows=gallery,report=report),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    write_csv(ART/'metrics.csv',table);write_csv(ART/'predictions.csv',flat);write_csv(ART/'reliability_bins.csv',bins)
    plot_results(report,arrays,yy,keep,bins)
    print(json.dumps(table,ensure_ascii=True),flush=True)


def plot_results(report,arrays,y,keep,bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[n for n in ['single_local_tenengrad_p75','logistic_basic','trees_basic','logistic_extended','trees_extended','neural'] if n in report['models']]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for n in names:
        p=arrays[n+'__p'][keep];fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=n);axes[1].plot(recall,precision,label=n)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[1].axhline(y.mean(),ls='--',c='gray')
    axes[0].set(xlabel='False positive rate',ylabel='Recall OK',title='Patient-held-out ROC',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Recall OK',ylabel='Precision OK',title='Patient-held-out precision / recall',xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(ART/'roc_pr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for n in names:
        for j,variant in enumerate(['raw','calibrated']):
            b=[b for b in bins if b['model']==n and b['kind']=='ok' and b['variant']==variant and b['count']]
            axes[j].plot([x['mean_probability'] for x in b],[x['observed_frequency'] for x in b],'o-',label=n)
    for j,ax in enumerate(axes):
        ax.plot([0,1],[0,1],'k--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean P(OK)',ylabel='Observed fraction OK',title=['Raw','Sigmoid calibrated'][j]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(ART/'calibration.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

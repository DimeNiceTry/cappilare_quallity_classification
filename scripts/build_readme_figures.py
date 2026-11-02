"""Build README figures from measured artifacts and original capillaroscopy frames."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/images'
ART=ROOT/'focus_classifier/artifacts'


def save(fig,name):
    fig.savefig(OUT/name,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((ART/'metrics.json').read_text(encoding='utf-8'))
    speed=json.loads((ART/'speed.json').read_text(encoding='utf-8'))
    audit=json.loads((ROOT/'research/audit_results.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    models=['logistic_basic','trees_extended','neural','constant']
    names=['4 признака + LR','43 признака + ExtraTrees','MobileNet + голова','Частота класса на train']
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(len(models));w=.36
    for shift,variant,title,color in [(-w/2,'raw','До калибровки','#c49a53'),(w/2,'calibrated','После калибровки','#248c76')]:
        values=[report['models'][m][variant]['ece_ok']['10'] for m in models]
        bars=ax.bar(x+shift,values,width=w,label=title,color=color)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=4,fontsize=10)
    ax.set(xticks=x,xticklabels=names,ylabel='ECE класса ok · 10 интервалов',title='Калибровка вероятностей на внешних пациентах',ylim=(0,.25))
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend();fig.tight_layout();save(fig,'ece_before_after.png')

    selections=[('neural','balanced','MobileNet · обычный'),('neural','precise','MobileNet · строгий'),
                ('trees_extended','balanced','ExtraTrees-43 · обычный'),('trees_extended','precise','ExtraTrees-43 · строгий'),
                ('logistic_basic','balanced','LR-4 · обычный')]
    fig,ax=plt.subplots(figsize=(11,5.2));y=np.arange(len(selections))
    good=[report['models'][m][mode]['tp'] for m,mode,_ in selections]
    bad=[report['models'][m][mode]['fp'] for m,mode,_ in selections]
    ax.barh(y,good,color='#248c76',label='Принятые хорошие');ax.barh(y,bad,left=good,color='#c86b63',label='Принятые плохие')
    for i,(g,b) in enumerate(zip(good,bad)):
        if g:ax.text(g/2,i,str(g),va='center',ha='center',color='white',weight='bold')
        if b:ax.text(g+b/2,i,str(b),va='center',ha='center',color='white',weight='bold')
        ax.text(g+b+3,i,f'{100*g/(g+b):.1f}% хороших',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=[v[2] for v in selections],xlabel='Число принятых кадров',title='Чистота отбора и сохранение хороших кадров',xlim=(0,220));ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'selection_tradeoff.png')

    keys=['logistic_basic','trees_basic','logistic_extended','trees_extended','neural']
    labels=['4 признака + LR','4 признака + ExtraTrees','43 признака + LR','43 признака + ExtraTrees','MobileNet + голова']
    fig,ax=plt.subplots(figsize=(11,4.8));y=np.arange(len(keys));w=.36
    for shift,key,title,color in [(-w/2,'end_to_end_ms_p50','PNG → решение','#426a86'),(w/2,'analysis_only_ms_p50','Готовый массив → решение','#248c76')]:
        values=[speed['models'][m][key] for m in keys];bars=ax.barh(y+shift,values,height=w,label=title,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=9)
    ax.set(yticks=y,yticklabels=labels,xlabel='Медиана, мс на кадр',title='CPU Intel i5-9300H · batch=1 · без GPU',xlim=(0,280));ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'cpu_latency.png')

    keys=['laplacian_raw','laplacian_smooth','tenengrad_smooth','local_tenengrad_p75']
    values=np.array([audit['scores'][k]['auc_ok_vs_not_ok'] for k in keys]);ci=np.array([audit['scores'][k]['patient_bootstrap_percentile_ci95'] for k in keys])
    fig,ax=plt.subplots(figsize=(10,4.6));y=np.arange(4)
    ax.errorbar(values,y,xerr=[values-ci[:,0],ci[:,1]-values],fmt='o',color='#2d7e77',capsize=5,markersize=8)
    ax.axvline(.5,color='gray',ls='--',label='Случайное ранжирование')
    for i,v in enumerate(values):ax.text(ci[i,1]+.015,i,f'{v:.3f}',va='center')
    ax.set(yticks=y,yticklabels=['Лапласиан исходный','Лапласиан после сглаживания','Tenengrad всего кадра','Локальный Tenengrad p75'],xlabel='AUROC · выше показатель → лучше качество',xlim=(.3,.8),title='Первые проверки: 790 исходных определённых меток');ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'raw_focus_auc.png')

    gallery=json.loads((ART/'gallery.json').read_text(encoding='utf-8'));by_file={r['file']:r for r in gallery['rows']}
    examples=[('14_08_2026_11_38_03_117.png','Согласовано: годен'),('12_08_2026_11_56_31_172.png','Согласовано: не годен · сильный шум'),('12_08_2026_12_36_05_6.png','Согласовано: не годен · смазывание')]
    fig,axes=plt.subplots(1,3,figsize=(15,5.4))
    for ax,(file,title) in zip(axes,examples):
        with Image.open(ROOT/'capillary_quality_1000/pictures'/file) as im:ax.imshow(np.array(im),interpolation='nearest')
        row=by_file[file];ax.axis('off')
        ax.set_title(title,fontsize=12,pad=13)
        ax.text(.5,-.04,f"P(ok): MobileNet {row['scores']['neural']['p']:.3f} · ExtraTrees-43 {row['scores']['trees_extended']['p']:.3f}",transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.suptitle('Конкретные примеры: шум и направленная текстура могут обманывать признаки',fontsize=15,y=.96)
    fig.tight_layout();save(fig,'photo_examples.png')
    (OUT/'sources.json').write_text(json.dumps(dict(metrics='focus_classifier/artifacts/metrics.json',speed='focus_classifier/artifacts/speed.json',original_feature_audit='research/audit_results.json',example_files=[v[0] for v in examples],predictions='patient-held-out OOF predictions, not final training-set predictions'),indent=2),encoding='utf-8')
    print('README figures prepared:',len(list(OUT.glob('*.png'))))


if __name__=='__main__':main()

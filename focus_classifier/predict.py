"""Predict new images using the separately saved final models."""
import argparse
import csv
from pathlib import Path
import cv2
import joblib
import numpy as np
from .features import extract
from .modeling import predict_bundle
from .train import ART


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='One native PNG or directory of PNG images')
    p.add_argument('--model',default='trees_basic',choices=['logistic_basic','trees_basic','logistic_extended','trees_extended','neural'])
    p.add_argument('--mode',choices=['balanced','precise'],default='balanced');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1)
    source='neural' if args.model=='neural' else 'features'
    b=joblib.load(ART/f'models_{source}.joblib')[args.model]
    files=sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    if not files:raise ValueError('No PNG images')
    if args.model=='neural':
        from .neural import load_backbone, embedding
        net=load_backbone()
    rows=[]
    for file in files:
        if args.model=='neural':x=embedding(net,file).reshape(1,-1)
        else:
            f=extract(file,basic_only=args.model.endswith('_basic'));x=np.array([[f[k] for k in b['features']]])
        prob,raw=predict_bundle(b,x);threshold=b['thresholds'][args.mode]
        rows.append(dict(file=str(file),model=args.model,p_ok=float(prob[0]),threshold=threshold,
                         predicted='ok' if prob[0]>=threshold else 'not_ok',
                         note='Final model trained on all consensus images; not OOF evaluation'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Saved {len(rows)} predictions: {args.out}')


if __name__=='__main__':main()

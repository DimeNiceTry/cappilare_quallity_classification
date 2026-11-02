"""Serial warm CPU latency; no model loading in per-frame timing."""
import json
import platform
import time
import joblib
import cv2
import numpy as np
from .features import from_gray
from .modeling import predict_bundle
from .neural import load_backbone, embedding_from_bgr
from .train import ROOT, ART


def main():
    cv2.setNumThreads(1)
    feature_models=joblib.load(ART/'models_features.joblib')
    models={n:feature_models[n] for n in ['logistic_basic','trees_basic','logistic_extended','trees_extended']}
    net=load_backbone()
    models['neural']=joblib.load(ART/'models_neural.joblib')['neural']
    files=sorted((ROOT/'capillary_quality_1000/pictures').glob('*.png'))
    rng=np.random.default_rng(20261102);sample=[files[i] for i in rng.choice(len(files),55,replace=False)]
    result={}
    for name,b in models.items():
        elapsed=[];pred_time=[];analysis_time=[];input_time=[]
        for path in sample:
            start=time.perf_counter()
            a=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR if name=='neural' else cv2.IMREAD_GRAYSCALE)
            decoded=time.perf_counter()
            if name=='neural':x=embedding_from_bgr(net,a).reshape(1,-1)
            else:
                f=from_gray(a,basic_only=name.endswith('_basic'));x=np.array([[f[k] for k in b['features']]])
            middle=time.perf_counter();predict_bundle(b,x);end=time.perf_counter()
            elapsed.append((end-start)*1000);pred_time.append((end-middle)*1000)
            analysis_time.append((end-decoded)*1000);input_time.append((decoded-start)*1000)
        elapsed=elapsed[5:];pred_time=pred_time[5:];analysis_time=analysis_time[5:];input_time=input_time[5:]
        result[name]=dict(n=50,end_to_end_ms_p50=float(np.median(elapsed)),end_to_end_ms_p95=float(np.percentile(elapsed,95)),
                          analysis_only_ms_p50=float(np.median(analysis_time)),analysis_only_ms_p95=float(np.percentile(analysis_time,95)),
                          png_read_decode_ms_p50=float(np.median(input_time)),head_only_ms_p50=float(np.median(pred_time)),throughput_sequential_fps=1000/float(np.mean(elapsed)))
        print(name,result[name],flush=True)
    report=dict(models=result,processor=platform.processor(),device='CPU',batch=1,opencv_threads=1,pytorch_threads=2,tree_inference_threads=1,
                note='Serial warm-process timing; PNG read/decode included; OS file cache may be warm; 5 warmup + 50 images each. Model load excluded. No concurrent training.')
    (ART/'speed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

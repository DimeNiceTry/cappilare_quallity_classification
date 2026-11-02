"""Separate image-only baseline: fixed ImageNet MobileNetV3 + learned head."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'focus_classifier/artifacts'


def load_backbone():
    torch.set_num_threads(2)
    torch.manual_seed(20261102)
    model=mobilenet_v3_small(weights=None)
    local=ART/'mobilenet_v3_small_imagenet.pt'
    if local.exists():
        model.load_state_dict(torch.load(local,map_location='cpu',weights_only=True))
    else:
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model.load_state_dict(weights.get_state_dict(progress=True,check_hash=True))
        torch.save(model.state_dict(),local)
    model.classifier=torch.nn.Identity()
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def tensor(path):
    a=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if a is None or a.shape[:2] != (768,1024):raise ValueError('Expected native 1024x768 image')
    return tensor_from_bgr(a)


def tensor_from_bgr(a):
    # Keep the entire field and aspect ratio; no centre crop.
    a=cv2.cvtColor(cv2.resize(a,(512,384),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    t=torch.from_numpy(a.transpose(2,0,1).copy()).float()/255.
    return (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def embedding(model,path):
    with torch.inference_mode():return model(tensor(path).unsqueeze(0)).numpy()[0]


def embedding_from_bgr(model,a):
    with torch.inference_mode():return model(tensor_from_bgr(a).unsqueeze(0)).numpy()[0]


def main():
    cv2.setNumThreads(1)
    files=[r['file'] for r in json.loads((ROOT/'capillary_quality_1000/frames.json').read_text(encoding='utf-8'))]
    model=load_backbone();features=[];timings=[]
    for i,file in enumerate(files):
        start=time.perf_counter();features.append(embedding(model,ROOT/'capillary_quality_1000/pictures'/file));timings.append(1000*(time.perf_counter()-start))
        if (i+1)%100==0:print(f'Neural embeddings {i+1}/{len(files)}',flush=True)
    np.savez_compressed(ART/'neural_embeddings.npz',files=np.array(files),features=np.array(features))
    meta=dict(architecture='MobileNetV3 Small',weights='ImageNet1K V1',frozen_backbone=True,
              learned_part='StandardScaler + L2 logistic head; inner sigmoid calibration',input_wh=[512,384],
              native_wh=[1024,768],crop=False,dimensions=len(features[0]),torch=torch.__version__,threads=2,
              extraction_p50_ms=float(np.median(timings[5:])),extraction_p95_ms=float(np.percentile(timings[5:],95)),
              note='Pretrained backbone never fitted on this dataset; only head and calibration fitted within patient folds.')
    (ART/'neural_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()

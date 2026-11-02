"""Fast installation and artifact checks without requiring the source dataset."""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import cv2,joblib,numpy,scipy,sklearn,torch,torchvision
    from PIL import Image
    required=['models_features.joblib','models_neural.joblib','mobilenet_v3_small_imagenet.pt',
              'gallery.json','features.csv','metrics.json']
    art=ROOT/'focus_classifier/artifacts'
    missing=[name for name in required if not (art/name).is_file()]
    if missing:raise FileNotFoundError('Missing artifacts: '+', '.join(missing))
    if sklearn.__version__!='1.5.2':raise RuntimeError('Use scikit-learn 1.5.2 for these saved models.')
    models=joblib.load(art/'models_features.joblib');models.update(joblib.load(art/'models_neural.joblib'))
    assert 'neural' in models and 'trees_extended' in models
    manifest=ROOT/'deployment/artifacts_sha256.json'
    if manifest.is_file():
        for rel,digest in json.loads(manifest.read_text(encoding='utf-8')).items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch: '+rel)
    print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,opencv=cv2.__version__,
                         sklearn=sklearn.__version__,torch=torch.__version__,torchvision=torchvision.__version__,
                         models=len(models),dataset_present=(ROOT/'capillary_quality_1000/pictures').is_dir()),ensure_ascii=True))


if __name__=='__main__':main()

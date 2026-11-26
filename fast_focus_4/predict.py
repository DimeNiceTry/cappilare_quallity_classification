"""Evaluate photos using the included four-feature models; no parent project needed."""
import argparse
import json
import sys
from pathlib import Path

# Works both as `python predict.py` inside the folder and `python -m fast_focus_4.predict`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_focus_4.workbench import (
    BASIC, MODEL_IDS, load_model, predict_files, verify_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify installation, artifacts, and both models')
    parser.add_argument('--input', type=Path, help='A native 1024x768 image or a folder')
    parser.add_argument('--out', type=Path, help='Output CSV, default: local_outputs/predictions.csv')
    parser.add_argument('--model', choices=MODEL_IDS, default='logistic_basic')
    parser.add_argument('--mode', choices=['balanced', 'precise'], default='balanced')
    parser.add_argument('--threshold', type=float, help='Optional manual threshold from 0 to 1')
    args = parser.parse_args()
    manifest = verify_artifacts()
    if args.check:
        import numpy as np
        import cv2
        import sklearn
        from fast_focus_4.core import from_gray, predict_bundle
        if sys.version_info[:2] not in [(3, 10), (3, 11)]:
            raise RuntimeError('Use Python 3.10 or 3.11')
        for name, actual in [('numpy', np.__version__), ('sklearn', sklearn.__version__), ('opencv', cv2.__version__)]:
            if actual != manifest[name]:
                raise RuntimeError(f'{name}: expected {manifest[name]}, found {actual}. Install requirements.txt')
        f = from_gray(np.zeros((768, 1024), dtype=np.uint8))
        for model in MODEL_IDS:
            p, _ = predict_bundle(load_model(model), np.array([[f[k] for k in BASIC]]))
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        print(json.dumps(dict(status='ok', models=list(MODEL_IDS), training=manifest['training'],
                              python=sys.version.split()[0], torch_required=False)))
    if args.input is None:
        if not args.check:
            parser.error('--input is required unless --check is used')
        return
    source = args.input.expanduser()
    if not source.exists():
        parser.error(f'Input does not exist: {source}')
    files = ([source] if source.is_file() else sorted(p for p in source.iterdir()
             if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}))
    if not files:
        parser.error('No supported photos found')
    result = predict_files(files, model=args.model, mode=args.mode, threshold=args.threshold)
    out = args.out or Path(__file__).resolve().parent/'local_outputs/predictions.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding='utf-8-sig')
    print(result[['file', 'p_ok', 'decision', 'known_training_image']].to_string(index=False))
    print(f'Saved: {out.resolve()}')


if __name__ == '__main__':
    main()

import csv
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .features import extract, VERSION, BASIC

ROOT = Path(__file__).resolve().parents[1]


def main():
    cv2.setNumThreads(1)
    dataset = ROOT / 'capillary_quality_1000'
    out = ROOT / 'focus_classifier' / 'artifacts'
    out.mkdir(exist_ok=True)
    original = json.loads((dataset/'frames.json').read_text(encoding='utf-8'))
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    expected = {r['file']: r['sha256'] for r in manifest['images_sha256']}
    old = {r['file']: r for r in csv.DictReader((ROOT/'research/focus_scores.csv').open(encoding='utf-8-sig'))}
    rows, times, diffs = [], [], []
    for i, r in enumerate(original):
        path = dataset/'pictures'/r['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[r['file']]:
            raise ValueError('Source image hash mismatch: '+r['file'])
        start = time.perf_counter()
        f = extract(path)
        times.append((time.perf_counter()-start)*1000)
        for name in BASIC:
            diffs.append(abs(f[name]-float(old[r['file']][name])))
        rows.append(dict(file=r['file'], sha256=digest, **f))
        if (i+1)%100 == 0:
            print(f'Features {i+1}/{len(original)}', flush=True)
    with (out/'features.csv').open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta = dict(version=VERSION,n=len(rows),features=list(f),basic_features=BASIC,
                extraction_ms_including_read_decode_p50=float(np.median(times)),
                extraction_ms_including_read_decode_p95=float(np.percentile(times,95)),
                max_absolute_difference_from_original_four_features=max(diffs),
                source_annotation_sha256=hashlib.sha256((dataset/'frames.json').read_bytes()).hexdigest())
    assert max(diffs)<1e-9
    (out/'features_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__ == '__main__':
    main()

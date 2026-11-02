"""Build four independent ZIP parts to distribute the original dataset via Releases."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'capillary_quality_1000';out=ROOT/'dist/data-v1';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    images=sorted(manifest['images_sha256'],key=lambda r:r['file'])
    assets=[]
    for part in range(4):
        path=out/f'capillary-data-part-{part+1:02d}.zip'
        subset=images[part*250:(part+1)*250]
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            if part==0:
                for file in source.iterdir():
                    if file.is_file():archive.write(file,'capillary_quality_1000/'+file.name)
            for row in subset:
                raw=(source/'pictures'/row['file']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Source hash mismatch: '+row['file'])
                archive.writestr('capillary_quality_1000/pictures/'+row['file'],raw)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hash_file(path)
        assets.append(dict(name=path.name,bytes=path.stat().st_size,sha256=digest,images=len(subset)))
        print(f'Prepared {path.name}: {path.stat().st_size} bytes',flush=True)
    index=dict(tag='data-v1',repository='DimeNiceTry/cappilare_quallity_classification',images=len(images),
               patients=manifest['patients'],videos=manifest['videos'],assets=assets,
               annotation_sha256=manifest['annotation_sha256'],note='Extract all four independent ZIP files into the repository root.')
    (out/'capillary-data-manifest.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    (ROOT/'deployment/dataset_release.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    print('Dataset archives complete.',flush=True)


def hash_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


if __name__=='__main__':main()

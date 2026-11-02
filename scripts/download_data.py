"""Download and verify the dataset Release; safe to rerun after interruption."""
import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--keep-archives',action='store_true');args=parser.parse_args()
    info=json.loads((ROOT/'deployment/dataset_release.json').read_text(encoding='utf-8'))
    cache=ROOT/'downloads';cache.mkdir(exist_ok=True)
    release_base=f"https://github.com/{info['repository']}/releases/download/{info['tag']}/"
    for asset in info['assets']:
        target=cache/asset['name']
        if not target.exists() or target.stat().st_size!=asset['bytes'] or digest(target)!=asset['sha256']:
            print('Downloading '+asset['name'],flush=True)
            request=urllib.request.Request(release_base+asset['name'],headers={'User-Agent':'capillary-quality-dataset'})
            with urllib.request.urlopen(request,timeout=180) as response,target.open('wb') as output:
                shutil.copyfileobj(response,output,length=8*1024*1024)
            if target.stat().st_size!=asset['bytes'] or digest(target)!=asset['sha256']:raise ValueError('Archive verification failed: '+asset['name'])
        print('Verified; extracting '+asset['name'],flush=True)
        with zipfile.ZipFile(target) as archive:
            for item in archive.infolist():
                relative=Path(item.filename)
                if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='capillary_quality_1000':
                    raise ValueError('Unexpected archive member')
                destination=(ROOT/relative).resolve()
                if ROOT.resolve() not in destination.parents:raise ValueError('Archive path escapes repository')
                if item.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source,destination.open('wb') as output:shutil.copyfileobj(source,output)
        if not args.keep_archives:target.unlink()
    manifest=json.loads((ROOT/'capillary_quality_1000/manifest.json').read_text(encoding='utf-8'))
    for row in manifest['images_sha256']:
        file=ROOT/'capillary_quality_1000/pictures'/row['file']
        if digest(file)!=row['sha256']:raise ValueError('Image verification failed: '+row['file'])
    print(f"Ready: {info['images']} original images; SHA-256 verified.",flush=True)


if __name__=='__main__':main()

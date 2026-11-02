"""Publish prepared dataset assets using Git Credential Manager, never a token file."""
import argparse
import hashlib
import http.client
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO='DimeNiceTry/cappilare_quallity_classification'
API='https://api.github.com/repos/'+REPO


def credentials():
    p=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],
                     input='protocol=https\nhost=github.com\nusername=DimeNiceTry\n\n',text=True,capture_output=True,
                     env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never'})
    values=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    if not values.get('password'):raise RuntimeError('GitHub authentication required; use Git Credential Manager.')
    return values['password']


def request(token,path,method='GET',data=None):
    raw=None if data is None else json.dumps(data).encode()
    req=urllib.request.Request(API+path,data=raw,method=method,headers={'Authorization':'Bearer '+token,
        'User-Agent':'capillary-project-packaging','Accept':'application/vnd.github+json','Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)


def upload(token,release,path):
    target=urllib.parse.urlsplit(release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(path.name))
    if target.scheme!='https' or target.hostname!='uploads.github.com':raise ValueError('Unexpected upload host')
    connection=http.client.HTTPSConnection(target.hostname,timeout=600)
    connection.putrequest('POST',target.path+'?'+target.query)
    for k,v in {'Authorization':'Bearer '+token,'User-Agent':'capillary-project-packaging','Accept':'application/vnd.github+json',
                'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)}.items():connection.putheader(k,v)
    connection.endheaders();sent=0;last=time.monotonic()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            connection.send(chunk);sent+=len(chunk)
            if time.monotonic()-last>20:
                print(f'{path.name}: {sent}/{path.stat().st_size} bytes',flush=True);last=time.monotonic()
    response=connection.getresponse();body=response.read();connection.close()
    if response.status not in (200,201):raise RuntimeError('Upload failed: HTTP '+str(response.status)+' '+body.decode()[:400])
    asset=json.loads(body)
    if asset['size']!=path.stat().st_size:raise ValueError('Remote asset size mismatch')
    print('Uploaded '+path.name,flush=True)
    return asset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft-only',action='store_true');args=parser.parse_args()
    token=credentials();index=json.loads((ROOT/'deployment/dataset_release.json').read_text())
    try:release=request(token,'/releases/tags/'+index['tag'])
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        release=request(token,'/releases','POST',dict(tag_name=index['tag'],name='Dataset v1 · 1000 capillaroscopy frames',draft=True,
            body='Original 1000 PNG images, 500 videos and 93 patient groups. Download all four independent ZIP parts and extract into the repository root, or run python scripts/download_data.py. SHA-256 checksums are in capillary-data-manifest.json. Historical labels and the newer consensus binary task are documented separately in the repository.'))
    folder=ROOT/'dist/data-v1'
    files=[folder/a['name'] for a in index['assets']]+[folder/'capillary-data-manifest.json']
    existing={a['name']:a for a in request(token,f"/releases/{release['id']}/assets?per_page=100")}
    for file in files:
        if file.name in existing:
            if existing[file.name]['size']!=file.stat().st_size:raise ValueError('Existing remote asset has a different size: '+file.name)
            print('Already uploaded '+file.name,flush=True);continue
        upload(token,release,file)
    if not args.draft_only:release=request(token,f"/releases/{release['id']}",'PATCH',dict(draft=False))
    result=dict(url=release['html_url'],id=release['id'],tag=release['tag_name'],draft=release['draft'])
    (ROOT/'dist/release_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

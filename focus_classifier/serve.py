"""Loopback comparison and in-memory upload inference; no directory listing."""
import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs
from .inference import InferenceEngine,MAX_UPLOAD

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'focus_classifier'
PORT=8767
ASSETS={'index.html','style.css','app.js','upload.html','upload.css','upload.js'}
ARTIFACTS={'gallery.json','metrics.json','metrics.csv','predictions.csv','speed.json','reliability_bins.csv','roc_pr.png','calibration.png','RESULTS_RU.md'}


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def json_response(self,value,status=200):
        self.respond(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'),'application/json',status=status)

    def do_GET(self):
        if not self.valid_host():
            self.send_error(403);return
        name=unquote(urlsplit(self.path).path)
        if name=='/api/health':
            self.json_response(dict(app='capillary-quality-comparison',version=2,inference=True));return
        if name=='/api/inference/config':
            try:self.json_response(dict(token=self.server.inference_token,models=self.server.engine.catalog(),max_bytes=MAX_UPLOAD))
            except Exception:self.json_response({'error':'Не удалось загрузить модели. Проверь локальные артефакты.'},500)
            return
        if name in ('/upload','/upload/'):name='/upload.html'
        if name=='/':name='/index.html'
        if name.startswith('/artifacts/') and name[len('/artifacts/'):] in ARTIFACTS:
            path=APP/'artifacts'/name[len('/artifacts/'):]
        elif name.lstrip('/') in ASSETS:
            path=APP/'static'/name.lstrip('/')
        elif name.startswith('/image/') and name[len('/image/'):] in self.server.image_files:
            path=ROOT/'capillary_quality_1000/pictures'/name[len('/image/'):]
        else:
            self.send_error(404);return
        if not path.is_file():self.send_error(404);return
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix=='.js':mime='text/javascript'
        self.respond(path.read_bytes(),mime)

    def do_POST(self):
        if not self.valid_host():self.send_error(403);return
        origin=self.headers.get('Origin')
        allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if origin is not None and origin not in allowed:self.json_response({'error':'Недопустимый источник запроса.'},403);return
        if not secrets.compare_digest(self.headers.get('X-Inference-Token',''),self.server.inference_token):
            self.json_response({'error':'Обнови страницу перед оценкой фотографий.'},403);return
        parsed=urlsplit(self.path)
        if parsed.path!='/api/inference/predict':self.send_error(404);return
        try:
            if self.headers.get('Transfer-Encoding'):raise ValueError('Неподдерживаемый способ передачи файла.')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_UPLOAD:
                self.json_response({'error':'Файл должен быть не больше 20 МБ.'},413);return
            raw=self.rfile.read(size)
            if len(raw)!=size:raise ValueError('Файл передан не полностью.')
            query=parse_qs(parsed.query)
            models=query.get('models',[''])[0].split(',')
            mode=query.get('mode',['precise'])[0]
            threshold=float(query['threshold'][0]) if 'threshold' in query else None
            result=self.server.engine.predict(raw,models,mode=mode,resize=query.get('resize',['0'])[0]=='1',manual_threshold=threshold)
            self.json_response(result)
        except (BrokenPipeError,ConnectionResetError):return
        except ValueError as error:self.json_response({'error':str(error)},400)
        except Exception:
            import traceback
            traceback.print_exc()
            self.json_response({'error':'Не удалось выполнить оценку. Подробности в server-error.log.'},500)

    def respond(self,body,mime,status=200):
        self.send_response(status);self.send_header('Content-Type',mime+('; charset=utf-8' if mime.startswith('text/') or mime=='application/json' else ''))
        self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'")
        self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(body)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=PORT)
    parser.add_argument('--open',action='store_true',help='Open upload page in the default browser')
    args=parser.parse_args()
    data=json.loads((APP/'artifacts/gallery.json').read_text(encoding='utf-8'))
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    server.image_files={r['file'] for r in data['rows']}
    server.engine=InferenceEngine()
    server.inference_token=secrets.token_urlsafe(32)
    print(f'http://127.0.0.1:{args.port}',flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/upload')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()

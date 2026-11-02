"""Local annotation server, Python standard library only."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from store import Conflict, Store

ROOT = Path(__file__).resolve().parent


def make_server(store, port):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, content_type="application/json; charset=utf-8", code=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def host_ok(self):
            return self.headers.get("Host", "") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.host_ok():
                return self.send({"error": "Invalid host"}, code=403)
            path = urlparse(self.path).path
            if path == "/api/health":
                return self.send(dict(app="capillary-cross-review", version=1))
            if path == "/api/state":
                return self.send(dict(**store.human_state(), token=token))
            if path == "/api/results":
                if not store.status()["complete"]:
                    return self.send({"error": "Сравнение откроется после завершения обеих очередей."}, code=409)
                return self.send(store.summary())
            if path == "/api/backup":
                # Contains only user's new answers, never the historical labels.
                return self.send(dict(**store.human_state(), type="human_review_backup"))
            if path.startswith("/image/"):
                row = store.by_id.get(path.removeprefix("/image/"))
                if row and row["assigned_to"] == "human":
                    return self.send((store.dataset / "pictures" / row["file"]).read_bytes(), "image/png")
                return self.send({"error": "Image not found"}, code=404)
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static:
                name, mime = static[path]
                return self.send((ROOT / "static" / name).read_bytes(), mime)
            return self.send({"error": "Not found"}, code=404)

        def do_POST(self):
            if not self.host_ok() or self.headers.get("X-Review-Token") != token:
                return self.send({"error": "Reload the page"}, code=403)
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self.send({"error": "Invalid origin"}, code=403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 20000:
                    raise ValueError("Invalid body length")
                data = json.loads(self.rfile.read(size))
                if self.path == "/api/review":
                    saved = store.save(data["id"], "human", data["label"], data.get("reasons", []), data.get("comment", ""), data.get("revision", 0))
                    return self.send(dict(saved=saved, progress=store.status()))
                if self.path == "/api/export":
                    return self.send(store.export(include_images=data.get("include_images") is True))
                return self.send({"error": "Not found"}, code=404)
            except Conflict as e:
                self.send({"error": str(e)}, code=409)
            except (ValueError, KeyError, TypeError) as e:
                self.send({"error": str(e)}, code=400)
            except Exception:
                self.send({"error": "Не удалось сохранить данные. Проверьте свободное место и перезапустите клиент."}, code=500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "capillary_quality_1000")
    parser.add_argument("--state", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    store = Store(args.data, args.state)
    server = make_server(store, args.port)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

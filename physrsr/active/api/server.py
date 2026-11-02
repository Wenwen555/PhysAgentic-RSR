"""Serial HTTP service host. Startup is explicit; no background experiment."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json


def serve(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if set(request) != {"method", "params"} or not isinstance(request["params"], dict):
                    raise ValueError("Expected method and params object")
                result = dict(result=service.call(request["method"], request["params"]))
            except Exception as exc:
                result = dict(error=f"{type(exc).__name__}: {exc}")
            body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer((host, port), Handler) as server:
        print(f"PhysRSR API listening on http://{host}:{port}", flush=True)
        try:
            server.serve_forever()
        finally:
            close = getattr(service, "close", None)
            if close:
                close()

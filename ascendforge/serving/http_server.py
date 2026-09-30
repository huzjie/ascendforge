"""Zero-dependency HTTP server exposing an OpenAI-compatible /v1/completions."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..backends import make_backend


class AscendForgeServer:
    def __init__(self, cfg=None, backend="mock", host="127.0.0.1", port=8000):
        self.cfg = cfg
        self.backend = make_backend(backend, cfg=cfg)
        self.host = host
        self.port = port
        self.model = "ascendforge-1.0"

    def complete(self, prompt, max_tokens=16):
        out = self.backend.generate(prompt)
        text = "ACCEPT" if out else "REJECT"
        return {
            "id": "cmpl-ascendforge",
            "object": "text_completion",
            "model": self.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    def start(self):
        server = self
        backend = self.backend
        model = self.model

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"data": [{"id": model, "object": "model"}]})
                elif self.path == "/health":
                    self._send(200, {"status": "ok"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/v1/completions":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    self._send(200, server.complete(body.get("prompt", ""),
                                                    body.get("max_tokens", 16)))
                else:
                    self._send(404, {"error": "not found"})

            def log_message(self, *args):
                pass

        httpd = HTTPServer((self.host, self.port), Handler)
        print(f"[serve] listening on http://{self.host}:{self.port} (backend={backend.name})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()


def serve(cfg=None, backend="mock", host="127.0.0.1", port=8000):
    AscendForgeServer(cfg, backend, host, port).start()

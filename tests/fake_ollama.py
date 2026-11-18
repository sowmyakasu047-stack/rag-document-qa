"""A small stand-in for the Ollama HTTP server, so OllamaLLM is tested over real HTTP.

It implements the two endpoints the code uses, as Ollama documents them:
``GET /api/tags`` (installed models) and ``POST /api/chat`` (non-streaming chat).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, models=("llama3.2:latest",), reply="Hello.", chat_status=200, chat_body=None):
        self.models = list(models)
        self.reply = reply
        self.chat_status = chat_status
        self.chat_body = chat_body  # raw bytes to send instead of the normal reply
        self.chat_requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, json.dumps({"models": [{"name": n} for n in outer.models]}).encode())
                else:
                    self._send(404, b'{"error": "not found"}')

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.chat_requests.append(json.loads(self.rfile.read(length)))
                if outer.chat_body is not None:
                    self._send(outer.chat_status, outer.chat_body)
                elif outer.chat_status != 200:
                    self._send(outer.chat_status, json.dumps({"error": "model 'x' not found"}).encode())
                else:
                    payload = {"message": {"role": "assistant", "content": outer.reply}, "done": True}
                    self._send(200, json.dumps(payload).encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

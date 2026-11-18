"""A minimal HTTP client that calls an ASGI app directly (no server, no httpx).

It builds real requests, including multipart bodies, and runs them through the
whole application stack, so tests exercise routing, form parsing and error handling.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

BOUNDARY = "----ragtestboundary"


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def multipart(fields: dict[str, Any]) -> tuple[bytes, str]:
    """Encode fields as multipart/form-data.

    A value of ``(filename, content_bytes)`` becomes a file part; a string becomes
    a plain text field.
    """
    parts: list[bytes] = []
    for name, value in fields.items():
        if isinstance(value, tuple):
            filename, content = value
            head = (
                f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            )
            parts.append(head.encode() + content + b"\r\n")
        else:
            head = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            parts.append(head.encode() + str(value).encode() + b"\r\n")
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={BOUNDARY}"


async def arequest(
    app,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    raw_body: bytes | None = None,
    files: dict[str, Any] | None = None,
    content_type: str | None = None,
) -> Response:
    headers: dict[str, str] = {}
    body = b""
    if json_body is not None:
        body, headers["content-type"] = json.dumps(json_body).encode(), "application/json"
    elif files is not None:
        body, headers["content-type"] = multipart(files)
    elif raw_body is not None:
        body = raw_body
    if content_type is not None:
        headers["content-type"] = content_type
    headers["content-length"] = str(len(body))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 50000),
    }

    pending = [{"type": "http.request", "body": body, "more_body": False}]
    started: dict[str, Any] = {}
    chunks: list[bytes] = []

    async def receive():
        if pending:
            return pending.pop()
        await asyncio.sleep(3600)  # the client never disconnects during the test
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            started["status"] = message["status"]
            started["headers"] = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return Response(started["status"], started["headers"], b"".join(chunks))


def request(app, method: str, path: str, **kwargs) -> Response:
    """Synchronous wrapper around :func:`arequest`."""
    return asyncio.run(arequest(app, method, path, **kwargs))

"""Lightweight local web UI server for the chonkie overlap refinery.

Serves the chunk-context workbench in ``ui/index.html`` and exposes a JSON API that runs the
real ``RecursiveChunker`` + ``OverlapRefinery`` from ``chonkie.refinery``. Standard library
only (plus chonkie itself). Start with ``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chonkie import RecursiveChunker  # noqa: E402
from chonkie.refinery import OverlapRefinery  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

MODES = ("token", "recursive")
METHODS = ("suffix", "prefix", "justified")
MAX_TEXT = 200_000


def refine_chunks(
    text: str,
    chunk_size: int = 400,
    context_size: float = 0.25,
    mode: str = "token",
    method: str = "suffix",
    merge: bool = False,
) -> dict[str, Any]:
    """Chunk ``text`` and attach overlap context; return the chunk breakdown.

    Raises ValueError with a clear message for invalid inputs.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT:
        raise ValueError(f"text exceeds {MAX_TEXT} characters")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 32:
        raise ValueError(f"chunk_size must be an integer >= 32, got {chunk_size!r}")
    if isinstance(context_size, bool) or not isinstance(context_size, (int, float)):
        raise ValueError(f"context_size must be a number, got {context_size!r}")
    if not 0.0 < float(context_size) < 1.0:
        raise ValueError(f"context_size must be in (0, 1), got {context_size}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")

    chunker = RecursiveChunker(chunk_size=int(chunk_size), min_characters_per_chunk=10)
    chunks = chunker(text)
    if not chunks:
        raise ValueError("text produced no chunks; provide more content")

    refinery = OverlapRefinery(
        tokenizer="character",
        context_size=float(context_size),
        mode=mode,
        method=method,
        merge=bool(merge),
        inplace=False,
    )
    refined = refinery.refine(chunks)

    out = []
    for index, chunk in enumerate(refined):
        context = getattr(chunk, "context", None) or ""
        out.append(
            {
                "index": index,
                "text": chunk.text,
                "token_count": int(chunk.token_count),
                "start_index": int(chunk.start_index),
                "end_index": int(chunk.end_index),
                "context": context,
                "context_len": len(context),
            }
        )

    total_context = sum(entry["context_len"] for entry in out)
    return {
        "chunk_count": len(out),
        "chunks": out,
        "mode": mode,
        "method": method,
        "merge": merge,
        "total_context_chars": total_context,
    }


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "modes": list(MODES), "methods": list(METHODS)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/refine":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = refine_chunks(
                    text=payload.get("text", ""),
                    chunk_size=payload.get("chunk_size", 400),
                    context_size=payload.get("context_size", 0.25),
                    mode=payload.get("mode", "token"),
                    method=payload.get("method", "suffix"),
                    merge=payload.get("merge", False),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the chunk-context workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8134)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Chunk-context workbench running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
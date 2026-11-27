"""Tests for the chunk-context workbench UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, refine_chunks

TEXT = "The quick brown fox jumps over the lazy dog. " * 12


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/refine",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_refine_endpoint_attaches_overlap_context(server):
    status, payload = _post(
        server,
        {"text": TEXT, "chunk_size": 120, "context_size": 0.25, "mode": "token", "method": "suffix"},
    )
    assert status == 200
    assert payload["chunk_count"] > 1
    assert payload["total_context_chars"] > 0
    # The final chunk has no next neighbor, so it carries no context.
    assert payload["chunks"][-1]["context_len"] == 0
    assert payload["chunks"][0]["context_len"] > 0


def test_result_is_json_serializable():
    result = refine_chunks(TEXT, chunk_size=120)
    json.dumps(result)  # must not raise


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["modes"] == ["token", "recursive"]
    assert health["methods"] == ["suffix", "prefix", "justified"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "workbench" in html and "/api/refine" in html


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"text": ""}, "text must be a non-empty string"),
        ({"text": TEXT, "chunk_size": 4}, "chunk_size must be an integer >= 32"),
        ({"text": TEXT, "context_size": 1.5}, "context_size must be in (0, 1)"),
        ({"text": TEXT, "mode": "magic"}, "mode must be one of"),
        ({"text": TEXT, "method": "magic"}, "method must be one of"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, kwargs, message):
    status, body = _post(server, kwargs)
    assert status == 400
    assert message in body["error"]

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


def test_index_has_loading_and_theme_states(server):
    """The page shows busy feedback during a run and follows the OS light/dark theme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert "prefers-color-scheme" in html, "page must follow the OS light/dark theme"


def test_oversized_declared_body_is_rejected_not_hung(server):
    """A lying Content-Length must get a fast 413, never park a handler thread."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(server)
    with socket.create_connection((u.hostname, u.port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/refine HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: 999999999\r\n\r\n{}"
        )
        first = sock.recv(4096).split(b"\r\n", 1)[0]
    assert b"413" in first, f"expected 413 for oversized body, got {first!r}"


def test_index_disables_heuristic_caching(server):
    """The UI shell must not be heuristically cached — stale HTML after edits was
    observed live because custom do_GET handlers send no cache validators."""
    resp = urllib.request.urlopen(server + "/")
    assert resp.headers.get("Cache-Control") == "no-cache"

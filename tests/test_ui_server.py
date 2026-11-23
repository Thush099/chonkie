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
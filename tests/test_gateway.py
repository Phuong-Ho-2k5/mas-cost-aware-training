import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from mas_demo.llm.gateway import OpenRouterGateway, GatewayError


@pytest.fixture
def upstream():
    state = {"count": 0, "status": 200, "content": {"status": "ABSTAIN", "hypotheses": [], "reason": "Not enough evidence"}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["count"] += 1
            state["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            envelope = {"choices": [{"message": {"content": json.dumps(state["content"])}}],
                        "usage": {"total_tokens": 5}}
            body = json.dumps({"error": state["error"]} if "error" in state else envelope).encode()
            try:
                if state.get("drip"):
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def test_gateway_sends_schema_and_parses_real_http(upstream):
    url, state = upstream
    result, metadata = OpenRouterGateway("dummy", base_url=url).complete({"epoch": 2})
    assert result.status == "ABSTAIN" and metadata["usage"]["total_tokens"] == 5
    assert state["request"]["response_format"]["type"] == "json_schema"


def test_unauthorized_does_not_retry(upstream):
    url, state = upstream
    state["status"] = 401
    with pytest.raises(GatewayError, match="401"):
        OpenRouterGateway("dummy", base_url=url).complete({})
    assert state["count"] == 1


def test_retryable_failure_has_bounded_attempts(upstream):
    url, state = upstream
    state["status"] = 429
    with pytest.raises(GatewayError):
        OpenRouterGateway("dummy", base_url=url, retry_delay=0).complete({})
    assert state["count"] == 2


def test_rate_limit_metadata_keeps_provider_message_without_api_key(upstream):
    url, state = upstream
    state["status"] = 429
    state["error"] = {"code": 429, "message": "Rate limit for dummy; try later"}
    with pytest.raises(GatewayError) as caught:
        OpenRouterGateway("dummy", base_url=url, retry_delay=0).complete({})
    errors = caught.value.metadata["upstream_errors"]
    assert len(errors) == 2 and errors[0]["status"] == 429
    assert errors[0]["message"] == "Rate limit for [REDACTED]; try later"
    assert "dummy" not in json.dumps(caught.value.metadata)


def test_schema_invalid_output_never_becomes_an_action(upstream):
    url, state = upstream
    state["content"] = {"command": "arbitrary shell"}
    with pytest.raises(GatewayError, match="schema"):
        OpenRouterGateway("dummy", base_url=url).complete({})


def test_absolute_deadline_cancels_a_slow_dripping_body(upstream):
    url, state = upstream
    state["drip"] = True
    started = time.monotonic()
    with pytest.raises(GatewayError, match="deadline"):
        OpenRouterGateway("dummy", base_url=url, timeout=0.5, deadline=0.3).complete({})
    assert time.monotonic() - started < 0.8

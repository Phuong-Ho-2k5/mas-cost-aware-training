import asyncio
import json
import threading
import time

import httpx
import pytest

pytest.importorskip(
    "fastapi", reason="Install the local-llm extra to test the optional serving service"
)


def request_body(schema=None, **overrides):
    return {
        "model": "test-qwen",
        "temperature": 0,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Return a result"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": schema
                or {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
        **overrides,
    }


class Engine:
    model_id = "test-qwen"
    info = {
        "model": model_id,
        "revision": "test-revision",
        "dtype": "float32",
        "device": "cpu",
    }

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.output = '{"ok":true}'
        self.finish_reason = "stop"
        self.active = self.peak = 0

    def generate(self, body, cancel):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if self.block:
                assert self.release.wait(5)
            return {
                "content": self.output,
                "finish_reason": self.finish_reason,
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
            }
        finally:
            self.active -= 1


def app_for(engine):
    from services.local_llm.server import create_app

    return create_app(lambda: engine)


def test_token_count_endpoint_does_not_run_generation():
    from fastapi.testclient import TestClient

    engine = Engine()
    engine.count_tokens = lambda body: 17
    with TestClient(app_for(engine)) as client:
        response = client.post("/v1/token-count", json=request_body())
    assert response.status_code == 200
    assert response.json() == {"prompt_tokens": 17, "context_tokens": 4096}
    assert engine.peak == 0


def test_models_health_and_valid_structured_response():
    from fastapi.testclient import TestClient

    engine = Engine()
    with TestClient(app_for(engine)) as client:
        assert client.get("/health").json()["ready"] is True
        assert client.get("/v1/models").json()["data"][0]["id"] == "test-qwen"
        data = client.post("/v1/chat/completions", json=request_body()).json()
        assert json.loads(data["choices"][0]["message"]["content"]) == {"ok": True}
        assert data["local_metadata"]["revision"] == "test-revision"
        assert data["usage"]["total_tokens"] == 10


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"model": "wrong-model"}, 404),
        ({"temperature": 1}, 422),
        ({"max_tokens": 99999}, 422),
        ({"stream": True}, 422),
        ({"response_format": {"type": "json_object"}}, 422),
        ({"messages": []}, 422),
        ({"extra_field": "ignored?"}, 422),
    ],
)
def test_unsupported_requests_fail_explicitly(overrides, code):
    from fastapi.testclient import TestClient

    with TestClient(app_for(Engine())) as client:
        response = client.post("/v1/chat/completions", json=request_body(**overrides))
        assert response.status_code == code
        assert "error" in response.json()


@pytest.mark.parametrize(
    "content,finish",
    [
        ('{"ok":true,"extra":1}', "stop"),
        ('{"ok":1}', "stop"),
        ('{"ok":true}', "length"),
        ('{"ok":', "length"),
    ],
)
def test_invalid_or_truncated_output_is_not_returned_as_success(content, finish):
    from fastapi.testclient import TestClient

    engine = Engine()
    engine.output, engine.finish_reason = content, finish
    with TestClient(app_for(engine)) as client:
        response = client.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_model_output"
        assert response.json()["error"]["raw_output"] == content


def test_numeric_and_reference_constraints_are_validated_after_generation():
    from fastapi.testclient import TestClient

    engine = Engine()
    engine.output = '{"value":2}'
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/unit"}},
        "required": ["value"],
        "additionalProperties": False,
        "$defs": {"unit": {"type": "number", "minimum": 0, "maximum": 1}},
    }
    with TestClient(app_for(engine)) as client:
        assert (
            client.post("/v1/chat/completions", json=request_body(schema)).status_code
            == 502
        )


def test_external_schema_references_are_rejected_without_fetching():
    from fastapi.testclient import TestClient

    with TestClient(app_for(Engine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json=request_body({"$ref": "https://example.com/schema"}),
        )
        assert response.status_code == 422


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/$defs/missing"},
        {"$defs": {"loop": {"$ref": "#/$defs/loop"}}, "$ref": "#/$defs/loop"},
    ],
)
def test_unresolved_and_recursive_schema_references_are_rejected(schema):
    from fastapi.testclient import TestClient

    with TestClient(app_for(Engine())) as client:
        response = client.post("/v1/chat/completions", json=request_body(schema))
        assert response.status_code == 422


def test_cancelled_client_does_not_release_worker_until_generation_returns():
    engine = Engine()
    engine.block = True
    app = app_for(engine)

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/v1/chat/completions", json=request_body())
                )
                while not engine.started.is_set():
                    await asyncio.sleep(0.01)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(
                    client.post("/v1/chat/completions", json=request_body())
                )
                await asyncio.sleep(0.05)
                third = await client.post("/v1/chat/completions", json=request_body())
                assert third.status_code == 429
                assert engine.active == engine.peak == 1
                engine.release.set()
                assert (await second).status_code == 200
                assert engine.peak == 1

    asyncio.run(exercise())


def test_generation_engine_checks_context_before_generating():
    from services.local_llm.generation import validate_token_budget

    validate_token_budget(100, 128, 4096)
    with pytest.raises(ValueError, match="context"):
        validate_token_budget(4000, 128, 4096)


@pytest.mark.parametrize(
    "failure,status,code",
    [
        (
            ValueError("Requested context exceeds 4096"),
            422,
            "invalid_context_or_schema",
        ),
        (TimeoutError("Generation deadline exceeded"), 504, "generation_timeout"),
        (RuntimeError("CUDA out of memory"), 503, "generation_failed"),
    ],
)
def test_generation_failures_are_structured_and_release_admission(
    failure, status, code
):
    from fastapi.testclient import TestClient

    engine = Engine()

    def fail(body, cancel):
        raise failure

    engine.generate = fail
    with TestClient(app_for(engine)) as client:
        response = client.post("/v1/chat/completions", json=request_body())
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert client.get("/health").json()["pending"] == 0


def test_real_http_rejected_generation_is_not_scored_as_transport_failure():
    import socket
    import uvicorn
    from mas_demo.contracts import TaskSpec
    from mas_demo.llm.gateway import OpenAICompatibleGateway
    from mas_demo.llm.benchmark import run_case

    engine = Engine()
    engine.output, engine.finish_reason = '{"status":', "length"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app_for(engine), log_level="error", access_log=False)
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        gateway = OpenAICompatibleGateway(
            "",
            model="test-qwen",
            mode="local",
            attempts=1,
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        result = run_case(
            {
                "id": "rejected",
                "role": "analysis",
                "scenario": "insufficient",
                "expected": "ABSTAIN",
            },
            TaskSpec(analysis_mode="local"),
            gateway,
        )
        assert result["raw_output"] == '{"status":'
        assert result["transport_ok"] is True and result["api_success"] is False
        assert result["schema_valid"] is False and result["correct"] is False
        # The invalid generation gets one output-repair call; both are charged.
        assert result["usage"]["total_tokens"] == 20
        assert len(result["calls"]) == 2
        assert result["repair_attempted"] and not result["repair_succeeded"]
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()

"""Local HTTP LLM provider shared by agent and training integration tests."""

import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from mas_demo.llm.gateway import VLLMGateway


@pytest.fixture
def provider(monkeypatch, tmp_path):
    # Real HTTP vLLM protocol; only this stand-in's process provenance is simulated.
    from mas_demo.llm import health
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "vllm")
    model = "Qwen/Qwen3.5-0.8B"
    state = {"requests": [], "responses": {}, "statuses": {},
             "token_context": 4096, "response_model": model, "served_models": [model]}
    runtime_path = tmp_path / "http-standin-runtime.json"
    runtime_path.write_text(json.dumps({"process": {}, "profile": {"model": model}}))
    monkeypatch.setenv("LLM_RUNTIME_MANIFEST", str(runtime_path))
    real_verify = health.verify_vllm_runtime
    default_path = Path(__file__).resolve().parents[1] / "tmp/vllm/runtime-manifest.json"

    def verify(gateway):
        if gateway.runtime_manifest and Path(gateway.runtime_manifest) in (runtime_path, default_path):
            return {"engine": "http-standin", "context_tokens": state["token_context"],
                    "structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True}}
        return real_verify(gateway)

    monkeypatch.setattr(health, "verify_vllm_runtime", verify)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state.get("health_status", 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "data": [
                            {"id": model} for model in state.get("served_models", [])
                        ]
                    }
                ).encode()
            )

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                if self.path == "/tokenize":
                    state.setdefault("tokenize_requests", []).append(request)
                    if "tokenize_status" in state or "tokenize_response" in state:
                        self.send_response(state.get("tokenize_status", 200))
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(state.get("tokenize_response", {})).encode())
                        return
                payload = json.loads(request["messages"][1]["content"])
                state.setdefault("token_requests", []).append(payload)
                extra = 0
                previous = payload.get("previous_response")
                if previous and not (
                    isinstance(previous, dict) and previous.get("stored_in")
                ):
                    extra = 1500
                count = 900 + 400 * len(payload.get("history", [])) + extra
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"count": count, "max_model_len": state["token_context"]}
                    ).encode()
                )
                return
            role = request["response_format"]["json_schema"]["name"]
            payload = json.loads(request["messages"][1]["content"])
            state["requests"].append((role, payload, request))
            defaults = {
                "training_diagnosis": {
                    "status": "DIAGNOSED",
                    "hypotheses": [
                        {
                            "hypothesis_id": "lr",
                            "cause": "Loss unstable",
                            "evidence_epochs": payload.get("evidence", {}).get(
                                "epochs", []
                            ),
                            "confidence": 0.6,
                        }
                    ],
                    "reason": "Try lower LR",
                },
                "training_plan": {
                    "proposals": [],
                    "verification_epochs": None,
                    "reason": "Prefer stronger LR reduction",
                },
                "training_evaluation": {
                    "decision": (
                        "PROMOTE" if payload.get("eligible_branches") else "KEEP_MAIN"
                    ),
                    "selected_branch": next(
                        iter(payload.get("eligible_branches", [])), "main"
                    ),
                    "reason": "Compared validation snapshots",
                },
            }
            if not payload.get("evidence", {}).get("epochs"):
                defaults["training_diagnosis"] = {"status": "ABSTAIN", "hypotheses": [], "reason": "No evidence"}
            content = state["responses"].get(role, defaults[role])
            self.send_response(state["statuses"].get(role, 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "id": "local-request",
                        "model": state.get("response_model", "local-model"),
                        "choices": [
                            {
                                "finish_reason": state.get("finish_reason", "stop"),
                                "message": {"content": json.dumps(content)},
                            }
                        ],
                        "usage": {"total_tokens": 12},
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = VLLMGateway(
        "local-key", base_url=f"http://127.0.0.1:{server.server_port}/v1", retry_delay=0,
        runtime_manifest=str(runtime_path),
    )
    yield gateway, state
    server.shutdown()
    server.server_close()
    thread.join()

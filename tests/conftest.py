"""Local HTTP LLM provider shared by agent and training integration tests."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from mas_demo.llm.gateway import OpenRouterGateway


@pytest.fixture
def provider():
    state = {"requests": [], "responses": {}, "statuses": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state.get("health_status", 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": model} for model in state.get("served_models", [])]}).encode())

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            role = request["response_format"]["json_schema"]["name"]
            payload = json.loads(request["messages"][1]["content"])
            state["requests"].append((role, payload, request))
            defaults = {
                "training_diagnosis": {"status": "DIAGNOSED", "hypotheses": [{
                    "hypothesis_id": "lr", "cause": "Loss unstable", "evidence_epochs": payload.get("evidence", {}).get("epochs", []),
                    "interventions": payload.get("allowed_interventions", []), "confidence": 0.6}], "reason": "Try lower LR"},
                "training_plan": {"trial_ids": [t["trial_id"] for t in payload.get("eligible_trials", [])][::-1],
                                  "reason": "Prefer stronger LR reduction"},
                "training_evaluation": {"decision": "PROMOTE" if payload.get("eligible_branches") else "KEEP_MAIN",
                                        "selected_branch": next(iter(payload.get("eligible_branches", [])), "main"),
                                        "reason": "Compared validation snapshots"},
            }
            content = state["responses"].get(role, defaults[role])
            self.send_response(state["statuses"].get(role, 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": "local-request", "model": "local-model",
                "choices": [{"finish_reason": state.get("finish_reason", "stop"),
                             "message": {"content": json.dumps(content)}}],
                "usage": {"total_tokens": 12}}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = OpenRouterGateway("local-key", base_url=f"http://127.0.0.1:{server.server_port}", retry_delay=0)
    yield gateway, state
    server.shutdown()
    server.server_close()
    thread.join()


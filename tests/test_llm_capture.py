import importlib.util
import json
from pathlib import Path
import sys

import pytest


def test_capture_fingerprint_changes_with_prompts_and_keeps_order(tmp_path):
    from mas_demo.llm.capture import fingerprint
    first, second = tmp_path / "prompt.txt", tmp_path / "fixture.json"
    first.write_text("initial"); second.write_text("[]")
    before = fingerprint([first, second], tmp_path)
    assert before == fingerprint([second, first], tmp_path)
    first.write_text("changed")
    assert fingerprint([first, second], tmp_path)["sha256"] != before["sha256"]


def test_capture_output_is_immutable(tmp_path):
    from mas_demo.llm.capture import new_capture_directory
    path = tmp_path / "new"
    assert new_capture_directory(path) == path.resolve()
    (path / "keep.txt").write_text("raw evidence")
    with pytest.raises(FileExistsError):
        new_capture_directory(path)
    assert (path / "keep.txt").read_text() == "raw evidence"


def test_three_role_probe_honors_explicit_model_and_output(provider, tmp_path, monkeypatch):
    from mas_demo.llm import settings
    upstream, state = provider
    path = Path(__file__).resolve().parents[1] / "scripts/check_local_agents.py"
    spec = importlib.util.spec_from_file_location("check_agent_cli", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "vllm")
    output = tmp_path / "probe"
    state["served_models"] = ["candidate-model"]
    state["response_model"] = "candidate-model"
    monkeypatch.setattr(sys, "argv", [str(path), "--model", "candidate-model", "--output", str(output)])
    module.main()
    assert len(state["requests"]) == 3
    assert all(row[2]["model"] == "candidate-model" for row in state["requests"])
    assert len((output / "responses.jsonl").read_text().splitlines()) == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] == "candidate-model" and manifest["code_fingerprint"]["sha256"]


def test_benchmark_distinguishes_initial_from_repaired_output(provider, monkeypatch):
    from mas_demo.llm.benchmark import run_case
    from mas_demo.contracts import TaskSpec
    gateway, state = provider
    original = gateway.complete
    def complete(payload):
        state["responses"]["training_diagnosis"] = {
            "status": "ABSTAIN", "hypotheses": [] if payload.get("repair_attempt") else [{
                "hypothesis_id": "h1", "cause": "Check", "confidence": 0.7, "evidence_epochs": [3, 4]}], "reason": "Observe"}
        return original(payload)
    monkeypatch.setattr(gateway, "complete", complete)
    record = run_case({"id": "repair", "role": "analysis", "scenario": "unstable", "expected": "ABSTAIN"}, TaskSpec(), gateway)
    assert record["initial_schema_valid"] is False
    assert record["initial_semantic_valid"] is False
    assert record["semantic_valid"] is True and record["repair_attempted"]


def test_quality_summary_keeps_error_latency_and_all_gate_requirements():
    from mas_demo.llm.benchmark import summarize
    rows = []
    for role in ("analysis", "planner", "evaluation"):
        for i in range(30):
            rows.append(dict(role=role, case_id=f"{role}-{i%10}", correct=True,
                schema_valid=True, semantic_valid=True, initial_schema_valid=True,
                initial_semantic_valid=True, transport_ok=True, latency_seconds=1,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                actual="NONEMPTY" if role=="planner" else "VALID",
                selected_horizon=1+i%2 if role=="planner" else None,
                novel_numeric_proposals=1 if role=="planner" else 0))
    report = summarize(rows)
    assert report["quality_thresholds_met"] is True
    assert report["token_usage"]["prompt_tokens"] == 900
    for record in rows[-6:]:
        record.update(semantic_valid=False,error_fallback=True,latency_seconds=130)
    report = summarize(rows)
    assert report["quality_thresholds_met"] is False
    assert report["latency_seconds"]["p95"] == 130
    assert report["roles"]["evaluation"]["error_fallback_rate"] == 0.2


def test_capture_accepts_empty_successful_vllm_health(provider, tmp_path, monkeypatch):
    import httpx
    from mas_demo.llm.capture import Capture
    gateway, state = provider
    original = httpx.Client.get
    def get(client, url, **kwargs):
        if url.endswith("/health"):
            return httpx.Response(200, content=b"", request=httpx.Request("GET", url))
        return original(client, url, **kwargs)
    monkeypatch.setattr(httpx.Client, "get", get)
    capture = Capture(tmp_path / "empty-health", gateway)
    assert capture.health == {}
    assert (capture.directory / "manifest.json").exists()


def test_pilot_accounts_for_zero_attempt_failed_invocations():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_local_training.py"
    spec = importlib.util.spec_from_file_location("pilot_accounting", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    events = [
        {"kind":"diagnosis","metadata":{"agent":"analysis","mode":"vllm","attempts":0,"latency_seconds":5,"error":"preflight"}},
        {"kind":"planning","metadata":{"agent":"planner","mode":"vllm","attempts":1,"latency_seconds":3}},
        {"kind":"diagnosis","metadata":{"agent":"analysis","mode":"disabled_baseline","attempts":0,"latency_seconds":0}},
    ]
    assert module.summarize_agent_costs(events) == {"llm_seconds":8,"llm_calls":2,"llm_http_attempts":1}


def test_failed_benchmark_report_keeps_capture_failed(provider, tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_local_agents.py"
    spec = importlib.util.spec_from_file_location("failed_bench_report", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gateway, state = provider
    state["served_models"] = [gateway.model]
    monkeypatch.setattr(module,"load_environment",lambda:None)
    monkeypatch.setattr(module,"create_gateway",lambda spec:gateway)
    monkeypatch.setattr(module,"load_cases",lambda path:[{"id":"a-one","role":"analysis"}])
    monkeypatch.setattr(module,"run_case",lambda *args:{"schema_valid":True,"correct":True})
    def fail(records):
        raise RuntimeError("report failed")
    monkeypatch.setattr(module,"summarize",fail)
    output=tmp_path/"failed-report"
    monkeypatch.setattr(sys,"argv",[str(path),"--rounds","1","--output",str(output)])
    with pytest.raises(RuntimeError,match="report failed"):
        module.main()
    assert json.loads((output/"manifest.json").read_text())["state"] == "failed"

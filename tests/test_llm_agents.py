"""Exercise real agent decisions through a local HTTP provider stand-in."""
import json

import pytest

from mas_demo.agents.analysis import diagnose
from mas_demo.agents import planner
from mas_demo.agents import evaluation
from mas_demo.contracts import (TaskSpec, DiagnosisReport, Hypothesis, EpochEvent,
                                EvidenceWindow, CheckpointRef, SegmentResult, Intervention)
from mas_demo.orchestration import run




def diagnosis():
    return DiagnosisReport(status="DIAGNOSED", mode="openrouter", hypotheses=[Hypothesis(
        hypothesis_id="lr", cause="unstable", evidence_epochs=[3, 4],
        interventions=[Intervention(parameter="learning_rate", operation="multiply", value=f) for f in [0.5, 0.1]], confidence=0.5)])


def metric(epoch=6, acc=0.5, loss=1.0, branch="main"):
    return EpochEvent(epoch=epoch, branch_id=branch, train_loss=1, val_loss=loss,
                      val_accuracy=acc, learning_rate=0.1, grad_norm=1)


def test_planner_uses_llm_order_and_role_schema(provider):
    gateway, state = provider
    trials, meta = planner.plan_with_agent(TaskSpec(), diagnosis(), [], 1, 2, gateway=gateway)
    assert [t.lr_factor for t in trials] == [0.1, 0.5]
    assert all(t.horizon == 2 for t in trials)
    assert meta["agent"] == "planner" and meta["usage"]["total_tokens"] == 12
    role, payload, request = state["requests"][0]
    assert role == "training_plan"
    assert request["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert payload["diagnosis"]["hypotheses"][0]["evidence_epochs"] == [3, 4]


@pytest.mark.parametrize("ids", [["unknown"], ["trial-2", "trial-2"], ["trial-1"]])
def test_planner_rejects_unknown_duplicate_and_cooldown_candidates(provider, ids):
    gateway, state = provider
    history = [{"factor": 0.5, "status": "FAILED", "round_id": 1, "gain": 0, "seconds": 2}]
    state["responses"]["training_plan"] = {"trial_ids": ids, "reason": "Choose"}
    trials, meta = planner.plan_with_agent(TaskSpec(analysis_mode="openrouter"), diagnosis(), history, 2, 2, gateway=gateway)
    assert trials == [] and "error" in meta


def test_planner_enforces_max_trials_and_allowlist(provider):
    gateway, state = provider
    spec = TaskSpec(analysis_mode="openrouter", max_trials=1)
    trials, _ = planner.plan_with_agent(spec, diagnosis(), [], 1, 2, gateway=gateway)
    assert len(trials) == 1 and trials[0].lr_factor == 0.1
    report = diagnosis()
    report.hypotheses[0].interventions = [Intervention(parameter="learning_rate", operation="multiply", value=0.99)]
    trials, meta = planner.plan_with_agent(spec, report, [], 1, 2, gateway=gateway)
    assert trials == [] and "error" in meta


def test_evaluator_uses_llm_choice_and_computes_gain(provider):
    gateway, state = provider
    trials = {"A": [metric(acc=0.6, loss=0.9, branch="A")],
              "B": [metric(acc=0.7, loss=0.8, branch="B")]}
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE", "selected_branch": "A", "reason": "Prefer A"}
    verdict, meta = evaluation.evaluate_with_agent(TaskSpec(), [metric()], trials, 6, gateway=gateway)
    assert verdict.selected_branch == "A"  # LLM chooses A; deterministic best would be B.
    assert verdict.gain == pytest.approx(0.1)
    assert meta["agent"] == "evaluation" and meta["model"] == "local-model"


@pytest.mark.parametrize("branch,events", [
    ("unknown", [metric(acc=0.7, loss=0.8)]),
    ("A", [metric(acc=0.7, loss=1.1)]),
    ("A", [metric(acc=0.501, loss=0.8)]),
    ("A", [metric(epoch=5, acc=0.7, loss=0.8)]),
])
def test_evaluator_cannot_promote_invalid_branch(provider, branch, events):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE", "selected_branch": branch, "reason": "Promote"}
    verdict, meta = evaluation.evaluate_with_agent(TaskSpec(analysis_mode="openrouter"), [metric()], {"A": events}, 6, gateway=gateway)
    assert verdict.decision == "INCONCLUSIVE" and verdict.selected_branch == "main"
    assert "error" in meta


def test_evaluator_can_keep_control_and_cannot_fabricate_gain(provider):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "KEEP_MAIN", "selected_branch": "main", "reason": "Keep"}
    verdict, _ = evaluation.evaluate_with_agent(TaskSpec(analysis_mode="openrouter"), [metric()],
        {"A": [metric(acc=0.7, loss=0.8)]}, 6, gateway=gateway)
    assert verdict.decision == "KEEP_MAIN"
    state["responses"]["training_evaluation"]["gain"] = 999
    verdict, meta = evaluation.evaluate_with_agent(TaskSpec(analysis_mode="openrouter"), [metric()],
        {"A": [metric(acc=0.7, loss=0.8)]}, 6, gateway=gateway)
    assert verdict.decision == "INCONCLUSIVE" and "error" in meta


def test_llm_evaluation_compares_exact_epoch_instead_of_latest_control(provider):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE", "selected_branch": "A", "reason": "Compare epoch 6"}
    verdict, meta = evaluation.evaluate_with_agent(TaskSpec(),
        [metric(acc=0.5), metric(epoch=7, acc=0.99)],
        {"A": [metric(acc=0.6, loss=0.9)]}, 6, gateway=gateway)
    assert verdict.selected_branch == "A" and verdict.gain == pytest.approx(0.1)
    assert "error" not in meta


def test_llm_evaluation_missing_snapshots_is_inconclusive(provider):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "INCONCLUSIVE", "selected_branch": "main", "reason": "Missing control"}
    verdict, meta = evaluation.evaluate_with_agent(TaskSpec(), [], {"A": [metric()]}, 6, gateway=gateway)
    assert verdict.decision == "INCONCLUSIVE" and "error" not in meta


@pytest.mark.parametrize("role", ["analysis", "planner", "evaluation"])
def test_agent_http_failure_never_falls_back_to_heuristic(provider, role):
    gateway, state = provider
    schema = {"analysis": "training_diagnosis", "planner": "training_plan", "evaluation": "training_evaluation"}[role]
    state["statuses"][schema] = 401
    spec = TaskSpec(analysis_mode="openrouter")
    if role == "analysis":
        evidence = EvidenceWindow(epochs=[3, 4], branch_id="main", status="READY", signals=["plateau"])
        result, meta = diagnose(spec, evidence, [], gateway=gateway)
        assert result.status == "ABSTAIN"
    elif role == "planner":
        result, meta = planner.plan_with_agent(spec, diagnosis(), [], 1, 2, gateway=gateway)
        assert result == []
    else:
        result, meta = evaluation.evaluate_with_agent(spec, [metric()], {"A": [metric(acc=0.8, loss=0.8)]}, 6, gateway=gateway)
        assert result.decision == "INCONCLUSIVE"
    assert meta["agent"] == role and meta["mode"] == "openrouter" and "401" in meta["error"]
    assert meta["attempts"] == 1 and meta["latency_seconds"] >= 0


def test_cli_environment_loading_preserves_process_key(tmp_path, monkeypatch):
    from mas_demo.cli import load_environment
    env_path = tmp_path/".env"
    env_path.write_text("OPENROUTER_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_environment(env_path)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "file-key"
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")
    load_environment(env_path)
    assert os.environ["OPENROUTER_API_KEY"] == "process-key"
    load_environment(tmp_path/"missing.env")
    assert os.environ["OPENROUTER_API_KEY"] == "process-key"


def test_heuristic_configuration_is_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TaskSpec(analysis_mode="heuristic")


def test_analysis_default_calls_api_instead_of_generating_a_hypothesis(provider):
    gateway, state = provider
    state["responses"]["training_diagnosis"] = {
        "status": "ABSTAIN", "hypotheses": [], "reason": "Provider declines diagnosis"}
    evidence = EvidenceWindow(epochs=[3, 4], branch_id="main", status="READY", signals=["plateau"])
    report, meta = diagnose(TaskSpec(), evidence, [], gateway=gateway)
    assert report.status == "ABSTAIN" and report.reason == "Provider declines diagnosis"
    assert meta["request_id"] == "local-request"


def test_cli_requires_key_before_starting_training(tmp_path, monkeypatch, capsys):
    from mas_demo import cli
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("Training started without API key"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["mas-demo", "run", "--config", "configs/smoke.yaml",
                                   "--output", str(tmp_path/"run")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not (tmp_path/"run").exists()


class ScriptedBackend:
    """Finite, persisted segments to isolate orchestrator behavior from ML noise."""
    def run(self, spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor=None, intervention=None):
        from pathlib import Path
        from mas_demo.io import sha256
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        begin = start.epoch if start else 0
        events = [metric(epoch=e, acc=0.7 if (intervention or lr_factor) else 0.5, loss=0.8 if (intervention or lr_factor) else 1.0,
                         branch=branch_id) for e in range(begin + 1, begin + epochs + 1)]
        path = output_dir/"checkpoint.pt"
        path.write_bytes(f"{branch_id}:{events[-1].epoch}".encode())
        checkpoint = CheckpointRef(path=str(path), sha256=sha256(path), epoch=events[-1].epoch,
            branch_id=branch_id, dataset_version="test", parent_sha256=start.sha256 if start else None)
        return SegmentResult(status="COMPLETE", checkpoint=checkpoint, best_checkpoint=checkpoint,
                             events=events, seconds=1)


def test_orchestrator_calls_all_three_agents_and_continues_selected_checkpoint(provider, tmp_path):
    gateway, state = provider
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [{
        "hypothesis_id": "lr", "cause": "plateau", "evidence_epochs": [1, 2],
        "interventions": [{"parameter": "learning_rate", "operation": "multiply", "value": f} for f in [0.5, 0.1]], "confidence": 0.6}], "reason": "Lower LR"}
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=16, val_samples=8, max_epochs=4, observation_interval=2, trial_horizon=2,
        analysis_mode="openrouter", budget_seconds=7, safety_margin=0,
        epoch_seconds_estimate=2, worker_overhead_seconds=0)
    directory = tmp_path/"run"
    result = run(spec, directory, backend=ScriptedBackend(), gateway=gateway)
    assert result["status"] == "COMPLETED"
    assert [r[0] for r in state["requests"]] == ["training_diagnosis", "training_plan", "training_evaluation"]
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    evaluation = next(e for e in events if e["kind"] == "evaluation")
    assert evaluation["verdict"]["decision"] == "PROMOTE"
    assert evaluation["result"]["comparison_epoch"] == 3  # Admission shortened horizon, without re-calling Planner.
    selected = evaluation["result"]["trials"][evaluation["verdict"]["selected_branch"]]
    assert result["checkpoint"]["parent_sha256"] == selected["checkpoint"]["sha256"]
    for kind in ("diagnosis", "planning", "evaluation"):
        record = next(e for e in events if e["kind"] == kind)
        assert record["metadata"]["usage"]["total_tokens"] == 12
        assert record["metadata"]["mode"] == "openrouter"


@pytest.mark.parametrize("role", ["analysis", "planner", "evaluation"])
def test_orchestrator_agent_failure_preserves_safe_continuation(provider, tmp_path, role):
    gateway, state = provider
    schemas = {"analysis": "training_diagnosis", "planner": "training_plan", "evaluation": "training_evaluation"}
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [{
        "hypothesis_id": "lr", "cause": "plateau", "evidence_epochs": [1, 2],
        "interventions": [{"parameter": "learning_rate", "operation": "multiply", "value": f} for f in [0.5, 0.1]], "confidence": 0.6}], "reason": "Lower LR"}
    state["statuses"][schemas[role]] = 401
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=16, val_samples=8, max_epochs=4, observation_interval=2, trial_horizon=1,
        analysis_mode="openrouter", epoch_seconds_estimate=1, worker_overhead_seconds=0)
    directory = tmp_path/"run"
    result = run(spec, directory, backend=ScriptedBackend(), gateway=gateway)
    assert result["status"] == "COMPLETED" and result["budget"]["reserved"] == 0
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    evaluations = [e for e in events if e["kind"] == "evaluation"]
    if role != "evaluation":
        assert not evaluations
        assert len(state["requests"]) == (1 if role == "analysis" else 2)
    else:
        record = evaluations[0]
        assert record["verdict"]["decision"] == "INCONCLUSIVE"
        assert result["checkpoint"]["parent_sha256"] == record["result"]["control"]["checkpoint"]["sha256"]
        assert "401" in record["metadata"]["error"]


@pytest.mark.parametrize("field,value", [("evidence_epochs", [999]), ("interventions", [{"parameter": "learning_rate", "operation": "multiply", "value": 0.99}])])
def test_analysis_rejects_fabricated_evidence_and_forbidden_factors(provider, field, value):
    gateway, state = provider
    response = diagnosis().model_dump(exclude={"mode"})
    response["hypotheses"][0][field] = value
    state["responses"]["training_diagnosis"] = response
    evidence = EvidenceWindow(epochs=[3, 4], branch_id="main", status="READY", signals=["plateau"])
    result, meta = diagnose(TaskSpec(analysis_mode="openrouter"), evidence, [], gateway=gateway)
    assert result.status == "ABSTAIN" and "error" in meta
    assert meta["usage"]["total_tokens"] == 12
    assert meta["output"] == response
    assert meta["validation_error"]["code"] == ("unavailable_epochs" if field == "evidence_epochs" else "forbidden_intervention")
    assert meta["validation_inputs"]["available_epochs"] == [3, 4]

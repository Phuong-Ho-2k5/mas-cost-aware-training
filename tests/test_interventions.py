"""Kiểm chứng can thiệp nhiều thông số bằng training thật và HTTP cục bộ."""
import json
import sqlite3
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from mas_demo import contracts
from mas_demo.contracts import TaskSpec, EvidenceWindow, EpochEvent
from mas_demo.data import prepare_data
from mas_demo.training.worker import train_segment
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.memory.store import MemoryStore
from mas_demo.orchestration import run


def action(parameter, value, operation="set"):
    return contracts.Intervention(parameter=parameter, operation=operation, value=value)


@pytest.mark.parametrize("parameter,value,operation", [
    ("momentum", 1, "set"), ("momentum", -0.1, "set"),
    ("weight_decay", -1, "set"), ("weight_decay", 2, "set"),
    ("grad_clip_norm", 0, "set"), ("grad_clip_norm", 101, "set"),
    ("learning_rate", 1, "multiply"), ("momentum", 0.5, "multiply"),
    ("learning_rate", 0.5, "set"), ("weight_decay", float("nan"), "set"),
    ("momentum", False, "set"), ("weight_decay", "0.5", "set"),
])
def test_invalid_intervention_is_rejected(parameter, value, operation):
    with pytest.raises(ValidationError):
        action(parameter, value, operation)


@pytest.mark.parametrize("parameter,initial,repaired", [
    ("momentum", 0.99, 0.5), ("weight_decay", 0.5, 0.001), ("grad_clip_norm", 0.01, 1.0),
])
def test_intervention_changes_only_target_and_persists_after_restore(tmp_path, parameter, initial, repaired):
    intervention = action(parameter, repaired)
    settings = {"momentum": 0.9, "weight_decay": 0.0, "grad_clip_norm": None, parameter: initial}
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=32, val_samples=16, batch_size=8, max_epochs=4,
        allowed_interventions=[intervention], **settings)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"first", 60)
    parent_bytes = Path(first.checkpoint.path).read_bytes()
    branch = train_segment(spec, first.checkpoint, 1, "trial", tmp_path/"branch", 60, intervention=intervention)
    control = train_segment(spec, first.checkpoint, 1, "main", tmp_path/"control", 60)
    assert branch.status == control.status == "COMPLETE"
    assert getattr(first.events[0], parameter) == initial
    for name, value in settings.items():
        assert getattr(branch.events[0], name) == (repaired if name == parameter else value)
        assert getattr(control.events[0], name) == value
    assert Path(first.checkpoint.path).read_bytes() == parent_bytes
    assert branch.checkpoint.parent_sha256 == control.checkpoint.parent_sha256 == first.checkpoint.sha256
    continued = train_segment(spec, branch.checkpoint, 1, "main", tmp_path/"continued", 60)
    assert getattr(continued.events[0], parameter) == repaired
    a = torch.load(branch.checkpoint.path, weights_only=False)
    b = torch.load(control.checkpoint.path, weights_only=False)
    assert any(not torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["effective_settings"][parameter] == repaired


def test_unallowlisted_intervention_never_changes_checkpoint(tmp_path):
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=16, val_samples=8, max_epochs=2)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"first", 60)
    before = Path(first.checkpoint.path).read_bytes()
    with pytest.raises(ValueError, match="allowlist"):
        train_segment(spec, first.checkpoint, 1, "bad", tmp_path/"bad", 60,
                      intervention=action("momentum", 0.5))
    assert Path(first.checkpoint.path).read_bytes() == before


def test_memory_preserves_legacy_lr_and_separates_equal_values(tmp_path):
    path = tmp_path/"memory.sqlite"
    memory = MemoryStore(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?)", ("ctx", 0.5, "REJECT", 0, 1, 1))
    momentum = action("momentum", 0.5)
    decay = action("weight_decay", 0.5)
    memory.record("ctx", momentum, "ACCEPT", 0.1, 2, 1)
    memory.record("ctx", decay, "FAILED", 0, 2, 1)
    rows = memory.retrieve("ctx", 2)
    assert len(rows) == 3
    assert {row["intervention"]["parameter"] for row in rows} == {"learning_rate", "momentum", "weight_decay"}


def test_legacy_memory_context_migration_uses_actual_lr_at_each_round(tmp_path):
    path = tmp_path/"memory.sqlite"
    store = MemoryStore(path)
    old_context = "synthetic:smallcnn:plateau"
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?)", (old_context, 0.5, "REJECT", 0, 1, 1))
        db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?)", (old_context, 0.1, "FAILED", 0, 1, 2))
    events = []
    for round_id, lr in [(1, 0.1), (2, 0.01)]:
        events.extend([
            {"kind": "evidence", "evidence": {"signals": ["plateau"], "recent_events": [{"learning_rate": lr}]}},
            {"kind": "diagnosis", "round": round_id}])
    log = tmp_path/"events.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    assert store.migrate_legacy_contexts(log, "synthetic") == 2
    context_at_high_lr = 'synthetic:smallcnn:plateau:{"grad_clip_norm": null, "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0}'
    context_at_low_lr = 'synthetic:smallcnn:plateau:{"grad_clip_norm": null, "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0}'
    assert [r["factor"] for r in store.retrieve(context_at_high_lr, 3)] == [0.5]
    assert [r["factor"] for r in store.retrieve(context_at_low_lr, 3)] == [0.1]
    assert len(store.retrieve(old_context, 3)) == 2  # Keep the source history intact.


def test_analysis_and_planner_use_generic_llm_actions(provider):
    gateway, state = provider
    intervention = action("weight_decay", 0.001)
    spec = TaskSpec(weight_decay=0.5, allowed_interventions=[intervention])
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [{
        "hypothesis_id": "decay", "cause": "Observed settings may regularize too strongly",
        "evidence_epochs": [1, 2], "interventions": [intervention.model_dump()], "confidence": 0.6}],
        "reason": "Verify against control"}
    evidence = EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY", signals=[], recent_events=[
        EpochEvent(epoch=e, branch_id="main", train_loss=1, val_loss=1,
            val_accuracy=0.5, learning_rate=0.01, grad_norm=1, weight_decay=0.5) for e in [1, 2]])
    report, metadata = diagnose(spec, evidence, [], gateway=gateway)
    assert report.status == "DIAGNOSED", report.reason
    trials, _ = plan_with_agent(spec, report, [], 1, 2, gateway=gateway)
    assert len(trials) == 1 and trials[0].intervention == intervention
    assert state["requests"][0][1]["schema_version"] == 2
    assert state["requests"][0][1]["evidence"]["recent_events"][-1]["weight_decay"] == 0.5
    assert metadata["usage"]["total_tokens"] == 12


@pytest.mark.parametrize("name,parameter,initial", [
    ("high_momentum", "momentum", 0.99), ("excessive_weight_decay", "weight_decay", 0.5),
    ("over_clipping", "grad_clip_norm", 0.01), ("mixed_faults", "learning_rate", 0.1),
])
def test_new_demo_configs_have_initial_faults_and_explicit_actions(name, parameter, initial):
    from mas_demo.cli import load_spec
    spec = load_spec(Path("configs")/(name+".yaml"))
    assert getattr(spec, parameter) == initial and spec.fault_epoch is None
    assert spec.allowed_interventions


def test_clipping_checkpoint_resume_matches_continuous_continuation(tmp_path):
    intervention = action("grad_clip_norm", 1)
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=32, val_samples=16, batch_size=8, max_epochs=4,
        grad_clip_norm=0.01, allowed_interventions=[intervention])
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"first", 60)
    branch = train_segment(spec, first.checkpoint, 1, "trial", tmp_path/"branch", 60, intervention=intervention)
    whole = train_segment(spec, branch.checkpoint, 2, "main", tmp_path/"whole", 60)
    part = train_segment(spec, branch.checkpoint, 1, "main", tmp_path/"part", 60)
    resumed = train_segment(spec, part.checkpoint, 1, "main", tmp_path/"resumed", 60)
    a = torch.load(whole.checkpoint.path, weights_only=False)
    b = torch.load(resumed.checkpoint.path, weights_only=False)
    assert whole.events[-1].grad_clip_norm == resumed.events[-1].grad_clip_norm == 1
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_subprocess_verification_applies_and_keeps_clipping(tmp_path):
    from mas_demo.execution.local import LocalBackend
    from mas_demo.control.budget import BudgetLedger
    from mas_demo.control.controller import admit_trials
    from mas_demo.control.verification import verify_round
    intervention = action("grad_clip_norm", 1)
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=16, val_samples=8, batch_size=8, max_epochs=3,
        grad_clip_norm=0.01, allowed_interventions=[intervention])
    prepare_data(spec)
    backend = LocalBackend()
    first = backend.run(spec, None, 1, "main", tmp_path/"first", 60)
    assert first.status == "COMPLETE", first.error
    trial = contracts.TrialSpec(trial_id="clip", intervention=intervention, horizon=1)
    ledger = BudgetLedger(300)
    admission = admit_trials([trial], ledger, 1, 59, 0, 0, 1)
    outcome = verify_round(spec, first.checkpoint, admission, ledger, backend, tmp_path/"round")
    assert outcome.control.status == outcome.trials["clip"].status == "COMPLETE"
    assert outcome.control.events[-1].grad_clip_norm == 0.01
    assert outcome.trials["clip"].events[-1].grad_clip_norm == 1
    resumed = backend.run(spec, outcome.trials["clip"].checkpoint, 1, "main", tmp_path/"resumed", 60)
    assert resumed.status == "COMPLETE" and resumed.events[-1].grad_clip_norm == 1
    request = json.loads((tmp_path/"round"/"clip"/"request.json").read_text())
    assert request["intervention"] == intervention.model_dump() and request["lr_factor"] is None


def test_legacy_checkpoint_keeps_optimizer_settings_and_disables_clipping(tmp_path):
    from mas_demo.training.checkpoint import save_checkpoint
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=16, val_samples=8, batch_size=8, max_epochs=2, weight_decay=0.1)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"first", 60)
    state = torch.load(first.checkpoint.path, weights_only=False)
    state.pop("effective_settings")
    legacy = save_checkpoint(tmp_path/"legacy.pt", state, "main", first.checkpoint.dataset_version)
    altered_defaults = spec.model_copy(update={"weight_decay": 0.5, "grad_clip_norm": 7})
    resumed = train_segment(altered_defaults, legacy, 1, "main", tmp_path/"resumed", 60)
    assert resumed.events[-1].weight_decay == 0.1 and resumed.events[-1].grad_clip_norm is None


def test_cooldown_filters_actions_independently_and_ignores_noop(provider, tmp_path):
    gateway, state = provider
    momentum = action("momentum", 0.5)
    decay = action("weight_decay", 0.5)
    store = MemoryStore(tmp_path/"memory.sqlite")
    store.record("ctx", momentum, "FAILED", 0, 1, 1)
    spec = TaskSpec(allowed_interventions=[momentum, decay])
    report = contracts.DiagnosisReport(status="DIAGNOSED", mode="openrouter", hypotheses=[contracts.Hypothesis(
        hypothesis_id="test", cause="Simulated hypothesis", evidence_epochs=[1, 2],
        interventions=[momentum, decay], confidence=0.5)])
    trials, _ = plan_with_agent(spec, report, store.retrieve("ctx", 2), 2, 1, gateway=gateway,
                               current_settings={"momentum": 0.99, "weight_decay": 0})
    assert [t.intervention for t in trials] == [decay]
    trials, _ = plan_with_agent(spec, report, [], 2, 1, gateway=gateway,
                               current_settings={"momentum": 0.5, "weight_decay": 0})
    assert [t.intervention for t in trials] == [decay]


def test_orchestrator_calls_analysis_without_detector_signal_and_promotes_non_lr(provider, tmp_path):
    from test_llm_agents import ScriptedBackend
    gateway, state = provider
    intervention = action("momentum", 0.5)

    class SettingsBackend(ScriptedBackend):
        """Scripted metrics isolate selection and continuation from ML randomness."""
        def __init__(self):
            self.settings_by_checkpoint = {}

        def run(self, spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor=None, intervention=None):
            settings = dict(self.settings_by_checkpoint[start.sha256]) if start else {
                "momentum": spec.momentum, "weight_decay": spec.weight_decay, "grad_clip_norm": spec.grad_clip_norm}
            if intervention:
                settings[intervention.parameter] = intervention.value
            result = super().run(spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor, intervention)
            # Descending control loss avoids every current detector's anomaly rules.
            for event in result.events:
                event.val_loss = 1.0 - event.epoch * 0.05 - (0.1 if intervention else 0)
                for parameter, value in settings.items():
                    setattr(event, parameter, value)
            self.settings_by_checkpoint[result.checkpoint.sha256] = settings
            return result

    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
        train_samples=16, val_samples=8, max_epochs=4, observation_interval=2, trial_horizon=1,
        momentum=0.99, allowed_interventions=[intervention], epoch_seconds_estimate=1,
        worker_overhead_seconds=0)
    directory = tmp_path/"run"
    result = run(spec, directory, backend=SettingsBackend(), gateway=gateway)
    assert result["status"] == "COMPLETED" and result["history"][-1]["momentum"] == 0.5
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["kind"] == "evidence")["evidence"]["signals"] == []
    verdict = next(e for e in events if e["kind"] == "evaluation")
    assert verdict["verdict"]["decision"] == "PROMOTE"
    assert [r[0] for r in state["requests"]] == ["training_diagnosis", "training_plan", "training_evaluation"]

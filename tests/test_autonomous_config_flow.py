import json
from pathlib import Path
import pytest
from mas_demo.contracts import (
    TaskSpec,
    TrainingConfig,
    ExperimentPolicy,
    EpochEvent,
    SegmentResult,
    CheckpointRef,
    ConfigPlanResponse,
    DiagnosisReport,
    Hypothesis,
    EvaluationResponse,
)
from mas_demo.configuration import config_hash, compile_plan, ProposalError
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_plan
from mas_demo.orchestration import run
from mas_demo.io import sha256


class ScriptedBackend:
    def __init__(self):
        self.calls = []

    def run(
        self,
        spec,
        start,
        epochs,
        branch_id,
        output_dir,
        budget_seconds,
        training_config=None,
    ):
        config = training_config or spec.training
        self.calls.append((branch_id, epochs, start.sha256 if start else None, config))
        begin = start.epoch if start else 0
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "checkpoint.pt"
        path.write_bytes(f"{branch_id}:{begin+epochs}".encode())
        manifest = json.loads(
            next(Path(spec.data_dir).glob("manifest-*.json")).read_text()
        )
        ref = CheckpointRef(
            path=str(path.resolve()),
            sha256=sha256(path),
            epoch=begin + epochs,
            branch_id=branch_id,
            dataset_version=manifest["version"],
            parent_sha256=start.sha256 if start else None,
            config_hash=config_hash(config),
        )
        events = [
            EpochEvent(
                epoch=e,
                branch_id=branch_id,
                train_loss=1,
                val_loss=0.8 if branch_id != "main" else 1,
                val_accuracy=0.7 if branch_id != "main" else 0.5,
                learning_rate=config.learning_rate,
                grad_norm=1,
            )
            for e in range(begin + 1, begin + epochs + 1)
        ]
        return SegmentResult(
            status="COMPLETE",
            checkpoint=ref,
            best_checkpoint=ref,
            events=events,
            seconds=1,
            training_config=config,
        )


class Gateway:
    def __init__(self, unaffordable=False):
        self.payloads = []
        self.unaffordable = unaffordable

    def complete(self, payload):
        self.payloads.append(("analysis", payload))
        epochs = payload["evidence"]["epochs"]
        abnormal = epochs[-1] == 1
        return (
            DiagnosisReport(
                status="DIAGNOSED" if abnormal else "ABSTAIN",
                mode="local",
                hypotheses=(
                    [
                        Hypothesis(
                            hypothesis_id="h1",
                            cause="Stalled learning",
                            evidence_epochs=epochs,
                            confidence=0.6,
                        )
                    ]
                    if abnormal
                    else []
                ),
                reason="Observe",
            ),
            {},
        )

    def structured(self, role, payload, response_type, instruction):
        self.payloads.append((role, payload))
        if role == "planner":
            count = 2 if "validation_errors" in payload else 3
            horizon = 1 if "validation_errors" in payload else 3
            proposals = [
                {
                    "proposal_id": f"p{i}",
                    "hypothesis_id": "h1",
                    "patch": {
                        "base_config_hash": payload["base_config_hash"],
                        "changes": {"learning_rate": v},
                        "reason": "Verify",
                        "evidence_epochs": payload["evidence"]["epochs"],
                    },
                    "expected_effect": "Improve",
                }
                for i, v in enumerate([0.027, 0.043, 0.067][:count])
            ]
            return (
                ConfigPlanResponse(
                    proposals=proposals, verification_epochs=horizon, reason="Compare"
                ),
                {},
            )
        return (
            EvaluationResponse(
                decision="PROMOTE",
                selected_branch=payload["eligible_branches"][0],
                reason="Measured gain",
            ),
            {},
        )


def test_three_trials_and_control_use_same_model_chosen_horizon_and_resume(tmp_path):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        policy=ExperimentPolicy(
            max_epochs=5,
            budget_seconds=100,
            epoch_seconds_estimate=1,
            worker_overhead_seconds=0,
            safety_margin=0,
        ),
    )
    backend, gateway = ScriptedBackend(), Gateway()
    result = run(spec, tmp_path / "run", backend=backend, gateway=gateway)
    assert result["status"] == "COMPLETED"
    verification = backend.calls[1:5]
    assert [c[0] for c in verification] == ["main", "trial-1", "trial-2", "trial-3"]
    assert all(c[1] == 3 for c in verification)
    assert len({c[2] for c in verification}) == 1
    assert result["active_config"]["learning_rate"] == 0.027
    assert backend.calls[-1][3].learning_rate == 0.027
    assert [
        p["evidence"]["epochs"][-1]
        for role, p in gateway.payloads
        if role == "analysis"
    ] == [1, 5]
    before = len(backend.calls)
    assert (
        run(spec, tmp_path / "run", resume=True, backend=backend, gateway=gateway)[
            "status"
        ]
        == "COMPLETED"
    )
    assert len(backend.calls) == before


def test_insufficient_budget_does_not_silently_shorten_or_drop_trials(tmp_path):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        policy=ExperimentPolicy(
            max_epochs=5,
            budget_seconds=10,
            epoch_seconds_estimate=1,
            worker_overhead_seconds=0,
            safety_margin=0,
        ),
    )
    backend, gateway = ScriptedBackend(), Gateway()
    result = run(spec, tmp_path / "run", backend=backend, gateway=gateway)
    planners = [p for role, p in gateway.payloads if role == "planner"]
    assert len(planners) == 2
    assert planners[1]["validation_errors"][0]["code"] == "INSUFFICIENT_BUDGET"
    assert [c[1] for c in backend.calls[1:4]] == [1, 1, 1]
    assert [c[0] for c in backend.calls[1:4]] == ["main", "trial-1", "trial-2"]
    assert result["budget"]["reserved"] == 0


def test_paused_run_resumes_and_active_manifest_contains_checkpoint_and_config(
    tmp_path,
):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "stop.flag").touch()
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        policy=ExperimentPolicy(
            max_epochs=2, epoch_seconds_estimate=1, worker_overhead_seconds=0
        ),
    )
    backend = ScriptedBackend()
    assert (
        run(spec, directory, backend=backend, gateway=Gateway())["status"] == "PAUSED"
    )
    assert not backend.calls
    (directory / "stop.flag").unlink()
    result = run(spec, directory, resume=True, backend=backend, gateway=Gateway())
    assert result["status"] == "COMPLETED"
    manifest = json.loads((directory / "state.json").read_text())
    assert manifest["checkpoint"]["config_hash"] == manifest["active_config_hash"]


def test_crash_during_worker_cannot_automatically_replay_unaccounted_work(tmp_path):
    class CrashBackend(ScriptedBackend):
        def run(self, *args, **kwargs):
            raise RuntimeError("Crash after admission")

    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        policy=ExperimentPolicy(
            max_epochs=2, epoch_seconds_estimate=1, worker_overhead_seconds=0
        ),
    )
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(spec, directory, backend=CrashBackend(), gateway=Gateway())
    backend = ScriptedBackend()
    assert (
        run(spec, directory, resume=True, backend=backend, gateway=Gateway())["status"]
        == "INTERRUPTED"
    )
    assert not backend.calls


def test_early_stopped_main_still_sends_its_final_epoch_to_analysis(tmp_path):
    from test_verification import InProcessBackend

    class ObserveOnly(Gateway):
        def complete(self, payload):
            self.payloads.append(("analysis", payload))
            return DiagnosisReport(status="ABSTAIN", mode="local", reason="Observe"), {}

    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        training=TrainingConfig(learning_rate=0.0, early_stop_patience=1),
        policy=ExperimentPolicy(
            max_epochs=8, epoch_seconds_estimate=10, worker_overhead_seconds=1
        ),
    )
    gateway = ObserveOnly()
    result = run(spec, tmp_path / "run", backend=InProcessBackend(), gateway=gateway)
    assert result["status"] == "EARLY_STOPPED"
    assert [p["evidence"]["epochs"][-1] for role, p in gateway.payloads] == [1, 2]

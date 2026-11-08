import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Intervention(Record):
    parameter: Literal["learning_rate", "momentum", "weight_decay", "grad_clip_norm"]
    operation: Literal["multiply", "set"]
    value: float = Field(strict=True)

    @model_validator(mode="after")
    def validate_action(self):
        if self.parameter == "learning_rate":
            valid = self.operation == "multiply" and 0 < self.value < 1
        else:
            limits = {"momentum": 0 <= self.value < 1, "weight_decay": 0 <= self.value <= 1,
                      "grad_clip_norm": 0 < self.value <= 100}
            valid = self.operation == "set" and limits[self.parameter]
        if not valid:
            raise ValueError("Invalid parameter, operation or intervention value")
        if self.value == 0:
            self.value = 0.0
        return self

    @property
    def key(self):
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))


class TaskSpec(Record):
    dataset: Literal["cifar10", "synthetic"] = "cifar10"
    data_dir: str = "data"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 11
    train_samples: int = Field(4096, gt=0, le=45000)
    val_samples: int = Field(1024, gt=0, le=5000)
    batch_size: int = Field(16, gt=0)
    learning_rate: float = Field(0.01, gt=0, le=10)
    momentum: float = Field(0.9, ge=0, lt=1)
    weight_decay: float = Field(0, ge=0, le=1)
    grad_clip_norm: float | None = Field(None, gt=0, le=100)
    max_epochs: int = Field(10, gt=0)
    observation_interval: int = Field(2, gt=0)
    window: int = Field(2, ge=2)
    trial_horizon: int = Field(2, gt=0)
    min_trial_horizon: int = Field(1, gt=0)
    max_trials: int = Field(2, ge=0, le=8)
    lr_factors: list[float] = Field(default_factory=lambda: [0.5, 0.1])
    allowed_interventions: list[Intervention] | None = None
    min_gain: float = Field(0.005, ge=0, le=1)
    budget_seconds: float = Field(900, gt=0)
    epoch_seconds_estimate: float = Field(20, gt=0)
    worker_overhead_seconds: float = Field(10, ge=0)
    safety_margin: float = Field(0.3, ge=0)
    unit_guard_seconds: float = Field(0.2, gt=0)
    analysis_mode: Literal["openrouter", "local"] = "openrouter"
    model_id: str = "xingchen-agi/xing4.0-29b-a4b:free"
    fault_epoch: int | None = Field(None, gt=0)
    fault_lr: float = Field(1.0, gt=0, le=10)

    @model_validator(mode="before")
    @classmethod
    def default_local_model(cls, value):
        if isinstance(value, dict) and value.get("analysis_mode") == "local" and "model_id" not in value:
            return {**value, "model_id": "Qwen/Qwen2.5-0.5B-Instruct"}
        return value

    @model_validator(mode="after")
    def validate_policy(self):
        if self.min_trial_horizon > self.trial_horizon:
            raise ValueError("min_trial_horizon exceeds trial_horizon")
        if not self.lr_factors or any(not 0 < x < 1 for x in self.lr_factors):
            raise ValueError("LR factors must be between 0 and 1")
        if len(set(self.lr_factors)) != len(self.lr_factors):
            raise ValueError("duplicate LR factors")
        if self.allowed_interventions is not None and len({a.key for a in self.allowed_interventions}) != len(self.allowed_interventions):
            raise ValueError("duplicate interventions")
        return self

    @property
    def intervention_allowlist(self):
        if self.allowed_interventions is not None:
            return self.allowed_interventions
        return [Intervention(parameter="learning_rate", operation="multiply", value=f) for f in self.lr_factors]


class DatasetManifest(Record):
    dataset: str
    seed: int
    train_indices: list[int]
    val_indices: list[int]
    version: str
    origin: Literal["real_subset", "real_full", "synthetic"]


class CheckpointRef(Record):
    path: str
    sha256: str
    epoch: int
    branch_id: str
    dataset_version: str
    parent_sha256: str | None = None


class EpochEvent(Record):
    epoch: int
    branch_id: str
    train_loss: float | None
    val_loss: float | None
    val_accuracy: float | None
    learning_rate: float
    grad_norm: float | None
    momentum: float | None = None
    weight_decay: float | None = None
    grad_clip_norm: float | None = None
    seconds: float = 0
    peak_vram_mb: float | None = None


class SegmentResult(Record):
    status: Literal["COMPLETE", "BUDGET_EXHAUSTED", "NUMERICAL_ERROR", "FAILED"]
    checkpoint: CheckpointRef | None = None
    best_checkpoint: CheckpointRef | None = None
    events: list[EpochEvent] = Field(default_factory=list)
    seconds: float = 0
    error: str | None = None


class EvidenceWindow(Record):
    epochs: list[int]
    branch_id: str
    status: Literal["READY", "INSUFFICIENT"]
    signals: list[str] = Field(default_factory=list)
    recent_events: list[EpochEvent] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class Hypothesis(Record):
    hypothesis_id: str
    cause: str
    evidence_epochs: list[int]
    interventions: list[Intervention]
    confidence: float = Field(ge=0, le=1)


class DiagnosisReport(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    mode: Literal["openrouter", "local"]
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    reason: str = ""


class PlanResponse(Record):
    trial_ids: list[str]
    reason: str


class EvaluationResponse(Record):
    decision: Literal["PROMOTE", "KEEP_MAIN", "INCONCLUSIVE"]
    selected_branch: str
    reason: str


class TrialSpec(Record):
    trial_id: str
    lr_factor: float | None = Field(None, gt=0, lt=1)
    intervention: Intervention | None = None
    horizon: int = Field(gt=0)
    hypothesis_id: str = ""

    @model_validator(mode="after")
    def normalize_action(self):
        if self.intervention is None:
            if self.lr_factor is None:
                raise ValueError("A trial needs an intervention")
            self.intervention = Intervention(parameter="learning_rate", operation="multiply", value=self.lr_factor)
        elif self.lr_factor is not None and self.intervention != Intervention(
                parameter="learning_rate", operation="multiply", value=self.lr_factor):
            raise ValueError("Conflicting trial intervention and legacy LR factor")
        if self.intervention.parameter == "learning_rate":
            self.lr_factor = self.intervention.value
        return self


class EvalVerdict(Record):
    decision: Literal["PROMOTE", "KEEP_MAIN", "INCONCLUSIVE"]
    selected_branch: str = "main"
    gain: float | None = None
    reason: str


class AdmissionBatch(Record):
    round_id: int
    horizon: int
    control_key: str | None = None
    continuation_key: str | None = None
    approved: list[TrialSpec] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    reservation_keys: dict[str, str] = Field(default_factory=dict)


class VerificationRoundResult(Record):
    comparison_epoch: int
    fork_ref: CheckpointRef
    control: SegmentResult
    trials: dict[str, SegmentResult] = Field(default_factory=dict)

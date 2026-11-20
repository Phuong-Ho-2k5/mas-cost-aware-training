"""Protocol v3: autonomous proposals and one shared verification horizon."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, StrictStr, StrictInt, StrictFloat


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TrainingConfig(Record):
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    learning_rate: float = Field(0.01, ge=0, strict=True)
    momentum: float = Field(0.9, ge=0, lt=1, strict=True)
    weight_decay: float = Field(0.0, ge=0, strict=True)
    grad_clip_norm: float | None = Field(None, gt=0, strict=True)
    batch_size: int = Field(16, gt=0, strict=True)
    gradient_accumulation: int = Field(1, gt=0, strict=True)
    beta1: float = Field(0.9, ge=0, lt=1, strict=True)
    beta2: float = Field(0.999, ge=0, lt=1, strict=True)
    epsilon: float = Field(1e-8, gt=0, strict=True)
    scheduler: Literal["none", "step", "cosine"] = "none"
    scheduler_step_size: int = Field(100, gt=0, strict=True)
    scheduler_gamma: float = Field(0.1, gt=0, strict=True)
    scheduler_t_max: int = Field(100, gt=0, strict=True)
    early_stop_patience: int | None = Field(None, gt=0, strict=True)


class TrainingPatch(Record):
    optimizer: Literal["sgd", "adam", "adamw"] = None
    learning_rate: float = Field(None, gt=0, strict=True)
    momentum: float = Field(None, ge=0, lt=1, strict=True)
    weight_decay: float = Field(None, ge=0, strict=True)
    grad_clip_norm: float | None = Field(None, gt=0, strict=True)
    batch_size: int = Field(None, gt=0, strict=True)
    gradient_accumulation: int = Field(None, gt=0, strict=True)
    beta1: float = Field(None, ge=0, lt=1, strict=True)
    beta2: float = Field(None, ge=0, lt=1, strict=True)
    epsilon: float = Field(None, gt=0, strict=True)
    scheduler: Literal["none", "step", "cosine"] = None
    scheduler_step_size: int = Field(None, gt=0, strict=True)
    scheduler_gamma: float = Field(None, gt=0, strict=True)
    scheduler_t_max: int = Field(None, gt=0, strict=True)
    early_stop_patience: int | None = Field(None, gt=0, strict=True)

    @model_validator(mode="after")
    def valid_nulls(self):
        for name in self.model_fields_set:
            if getattr(self, name) is None and name not in (
                "grad_clip_norm",
                "early_stop_patience",
            ):
                raise ValueError(f"{name} cannot be null")
        return self


class ExperimentPolicy(Record):
    max_epochs: int = Field(10, gt=0, strict=True)
    max_trials: int = Field(3, ge=0, le=8, strict=True)
    budget_seconds: float = Field(900, gt=0)
    min_gain: float = Field(0.005, ge=0, le=1)
    epoch_seconds_estimate: float = Field(20, gt=0)
    worker_overhead_seconds: float = Field(10, ge=0)
    safety_margin: float = Field(0.3, ge=0)
    cooldown_rounds: int = Field(2, ge=0, strict=True)


class TaskSpec(Record):
    schema_version: Literal[3] = 3
    dataset: Literal["cifar10", "synthetic", "blobs"] = "cifar10"
    data_dir: str = "data"
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    seed: int = 11
    train_samples: int = Field(4096, gt=0, le=45000)
    val_samples: int = Field(1024, gt=0, le=5000)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    policy: ExperimentPolicy = Field(default_factory=ExperimentPolicy)
    observation_interval: Literal[1] = 1
    window: int = Field(5, ge=1, le=8)
    unit_guard_seconds: float = Field(0.2, gt=0)
    analysis_mode: Literal["vllm"] = "vllm"
    model_id: str = "Qwen/Qwen3.5-0.8B"

    @model_validator(mode="before")
    @classmethod
    def blobs_sample_defaults(cls, values):
        if isinstance(values, dict) and values.get("dataset") == "blobs":
            values = dict(values)
            values.setdefault("train_samples", 700)
            values.setdefault("val_samples", 300)
        return values

    @model_validator(mode="after")
    def validate_blobs_split(self):
        if self.dataset == "blobs" and (
            self.train_samples > 700 or self.val_samples > 300
        ):
            raise ValueError(
                "Blobs uses the fixed AUTOTRAINER split: at most 700 train / 300 validation"
            )
        return self

    @property
    def max_epochs(self):
        return self.policy.max_epochs

    @property
    def budget_seconds(self):
        return self.policy.budget_seconds

    @property
    def min_gain(self):
        return self.policy.min_gain

    @property
    def max_trials(self):
        return self.policy.max_trials


class DatasetManifest(Record):
    dataset: str
    seed: int
    train_indices: list[int]
    val_indices: list[int]
    version: str
    origin: Literal["real_subset", "real_full", "synthetic", "generated_toy"]
    data_recipe: str | None = None
    generator_version: str | None = None


class CheckpointRef(Record):
    path: str
    sha256: str
    epoch: int
    branch_id: str
    dataset_version: str
    parent_sha256: str | None = None
    config_hash: str | None = None


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
    config_hash: str | None = None
    samples_seen: int = 0
    optimizer_steps: int = 0


class SegmentResult(Record):
    status: Literal[
        "COMPLETE", "EARLY_STOPPED", "BUDGET_EXHAUSTED", "NUMERICAL_ERROR", "FAILED"
    ]
    checkpoint: CheckpointRef | None = None
    best_checkpoint: CheckpointRef | None = None
    events: list[EpochEvent] = Field(default_factory=list)
    seconds: float = 0
    error: str | None = None
    training_config: TrainingConfig | None = None
    transition: dict = Field(default_factory=dict)


class EvidenceWindow(Record):
    epochs: list[int]
    branch_id: str
    status: Literal["READY", "INSUFFICIENT"]
    signals: list[str] = Field(default_factory=list)
    recent_events: list[EpochEvent] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class AnalysisHypothesis(Record):
    hypothesis_id: str
    rank: int = Field(ge=1, le=3, strict=True)
    cause: str
    evidence_epochs: list[int]
    confidence: float = Field(ge=0, le=1)


Hypothesis = AnalysisHypothesis


class ObservationFact(Record):
    fact_id: str
    metric: Literal["train_loss", "val_loss", "val_accuracy", "grad_norm"]
    kind: Literal["value", "step_delta", "window_delta", "pattern"]
    epochs: list[int]
    value: float | None = None
    direction: Literal["up", "down", "flat"] | None = None
    pattern: Literal["insufficient", "incomplete", "mixed", "nondecreasing", "nonincreasing", "flat"] | None = None

    @model_validator(mode="after")
    def consistent_fact(self):
        if not self.epochs or len(self.epochs) != len(set(self.epochs)):
            raise ValueError("Fact requires distinct evidence epochs")
        if self.kind == "pattern":
            if self.pattern is None or self.value is not None or self.direction is not None:
                raise ValueError("Pattern fact requires only a pattern")
        elif self.value is None or self.pattern is not None:
            raise ValueError("Numeric fact requires a value and no pattern")
        elif self.kind == "value":
            if len(self.epochs) != 1 or self.direction is not None:
                raise ValueError("Value fact has one epoch and no direction")
        elif len(self.epochs) != 2 or self.direction is None:
            raise ValueError("Delta fact requires two epochs and a direction")
        return self


class DetectionResponse(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    observation_ids: list[str] = Field(max_length=4)
    reason: str = Field(min_length=1, max_length=240)


class ConfigFact(Record):
    field: str
    value: StrictStr | StrictInt | StrictFloat | None
    active: bool


class ConfigClaim(Record):
    field: str
    observed_value: StrictStr | StrictInt | StrictFloat | None
    claim: Literal["active_effect", "missing_effect", "setting"]


class GeneratedHypothesis(AnalysisHypothesis):
    hypothesis_id: str = Field(min_length=1, max_length=48)
    cause: str = Field(min_length=1, max_length=160)
    mechanism: str = Field(min_length=1, max_length=120)
    observation_ids: list[str] = Field(min_length=1, max_length=4)
    config_claims: list[ConfigClaim] = Field(max_length=4)


class HypothesesResponse(Record):
    hypotheses: list[GeneratedHypothesis] = Field(min_length=3, max_length=3)
    reason: str = Field(min_length=1, max_length=160)


class EpochLogDetectionResponse(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    evidence_epochs: list[int] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=240)


class EpochLogHypothesis(AnalysisHypothesis):
    hypothesis_id: str = Field(min_length=1, max_length=48)
    cause: str = Field(min_length=1, max_length=160)
    mechanism: str = Field(min_length=1, max_length=120)
    config_claims: list[ConfigClaim] = Field(max_length=4)


class EpochLogHypothesesResponse(Record):
    hypotheses: list[EpochLogHypothesis] = Field(min_length=3, max_length=3)
    reason: str = Field(min_length=1, max_length=160)


class DiagnosisReport(Record):
    status: Literal["DIAGNOSED", "ABSTAIN"]
    mode: Literal["vllm"]
    hypotheses: list[AnalysisHypothesis] = Field(default_factory=list)
    reason: str = ""


class ConfigPatch(Record):
    base_config_hash: str
    changes: TrainingPatch
    reason: str
    evidence_epochs: list[int]


class ConfigProposal(Record):
    proposal_id: str
    hypothesis_id: str
    patch: ConfigPatch
    expected_effect: str


class ConfigPlanResponse(Record):
    proposals: list[ConfigProposal]
    verification_epochs: int | None = Field(ge=1, strict=True)
    reason: str

    @model_validator(mode="after")
    def shared_horizon(self):
        if bool(self.proposals) != (self.verification_epochs is not None):
            raise ValueError(
                "Nonempty proposals require verification_epochs; abstention requires null"
            )
        return self


class CompiledProposal(Record):
    trial_id: str
    proposal_id: str
    hypothesis_id: str
    base_config_hash: str
    training_config: TrainingConfig
    config_hash: str
    diff: dict
    evidence_epochs: list[int]
    hypothesis_rank: int | None = Field(None, ge=1, le=3, strict=True)


class CompiledPlan(Record):
    candidates: list[CompiledProposal]
    verification_epochs: int | None


class EvaluationResponse(Record):
    decision: Literal["PROMOTE", "KEEP_MAIN", "INCONCLUSIVE"]
    selected_branch: str
    reason: str


class EvalVerdict(EvaluationResponse):
    selected_branch: str = "main"
    gain: float | None = None


class TrialCostQuote(Record):
    trial_id: str
    proposal_id: str
    hypothesis_id: str
    hypothesis_rank: int = Field(ge=1, le=3, strict=True)
    epoch_seconds: float = Field(gt=0)
    reserved_seconds: float = Field(gt=0)


class PlanCostQuote(Record):
    horizon: int = Field(gt=0, strict=True)
    control_seconds: float = Field(gt=0)
    continuation_seconds: float = Field(ge=0)
    trial_costs: dict[str, TrialCostQuote]
    all_trials_total_seconds: float = Field(gt=0)


class SkippedTrial(Record):
    trial_id: str
    proposal_id: str
    hypothesis_id: str
    hypothesis_rank: int = Field(ge=1, le=3, strict=True)
    estimated_seconds: float = Field(gt=0)
    code: Literal["SKIPPED_BUDGET", "SKIPPED_POLICY", "SKIPPED_COOLDOWN"]
    reason: str


class AdmissionBatch(Record):
    round_id: int
    horizon: int
    control_key: str | None
    approved: list[CompiledProposal]
    reservation_keys: dict[str, str]
    continuation_key: str | None = None
    skipped: list[SkippedTrial] = Field(default_factory=list)
    cost_quote: PlanCostQuote | None = None
    budget_available_before: float | None = None
    reserved_total_seconds: float = 0


class VerificationRoundResult(Record):
    comparison_epoch: int
    fork_ref: CheckpointRef
    control: SegmentResult
    trials: dict[str, SegmentResult] = Field(default_factory=dict)

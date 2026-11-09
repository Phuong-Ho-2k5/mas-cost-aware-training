from typing import Protocol
from mas_demo.contracts import TaskSpec, CheckpointRef, SegmentResult, TrainingConfig


class ExecutionBackend(Protocol):
    def run(
        self,
        spec: TaskSpec,
        start: CheckpointRef | None,
        epochs: int,
        branch_id: str,
        output_dir,
        budget_seconds: float,
        training_config: TrainingConfig | None = None,
    ) -> SegmentResult: ...

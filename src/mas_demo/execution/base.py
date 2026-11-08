from typing import Protocol
from mas_demo.contracts import TaskSpec, CheckpointRef, SegmentResult, Intervention


class ExecutionBackend(Protocol):
    """Future Slurm implementation uses the same finite-segment contract."""

    def run(self, spec: TaskSpec, start: CheckpointRef | None, epochs: int,
            branch_id: str, output_dir, budget_seconds: float,
            lr_factor: float | None = None, intervention: Intervention | None = None) -> SegmentResult: ...

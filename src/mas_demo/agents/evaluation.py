import os
from mas_demo.contracts import EvalVerdict, EvaluationResponse
from mas_demo.llm.gateway import OpenRouterGateway, GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway


def evaluate_with_agent(spec, control, trials, comparison_epoch, gateway=None):
    def snapshot(events):
        return next((e for e in events if e.epoch == comparison_epoch and
                     e.val_accuracy is not None and e.val_loss is not None), None)

    base = snapshot(control)
    snapshots = {name: snapshot(events) for name, events in trials.items()}
    eligible = {name: e for name, e in snapshots.items() if base is not None and e is not None
                and e.val_accuracy - base.val_accuracy >= spec.min_gain and e.val_loss <= base.val_loss}
    metadata = {}
    try:
        gateway = gateway or create_gateway(spec)
        response, metadata = gateway.structured("evaluation", {
            "schema_version": 2, "comparison_epoch": comparison_epoch, "min_gain": spec.min_gain,
            "control": base.model_dump() if base else None,
            "trials": {name: e.model_dump() if e else None for name, e in snapshots.items()},
            "eligible_branches": list(eligible)}, EvaluationResponse,
            "Compare the provided validation snapshots at comparison_epoch. "
            "Return decision (PROMOTE, KEEP_MAIN, or INCONCLUSIVE), selected_branch, and reason. "
            "PROMOTE may select only an eligible_branches ID. Accuracy gain must meet min_gain "
            "and val_loss must not exceed control. Otherwise select main. "
            "If control or all trial snapshots are missing, return INCONCLUSIVE. Do not invent metrics or gain.")
        metadata.update(output=response.model_dump(), reason=response.reason)
        if response.decision == "PROMOTE":
            if response.selected_branch not in eligible:
                raise GatewayError("Evaluation tried to promote an ineligible branch")
            return EvalVerdict(**response.model_dump(),
                gain=eligible[response.selected_branch].val_accuracy - base.val_accuracy), metadata
        if response.selected_branch != "main":
            raise GatewayError("Non-promotion must select main")
        if response.decision == "KEEP_MAIN" and (base is None or not any(e is not None for e in snapshots.values())):
            raise GatewayError("No comparable snapshots; evaluation must be inconclusive")
        return EvalVerdict(**response.model_dump(), gain=0 if response.decision == "KEEP_MAIN" else None), metadata
    except GatewayError as error:
        return EvalVerdict(decision="INCONCLUSIVE", reason=str(error)), failure_metadata(error, "evaluation", spec, metadata)

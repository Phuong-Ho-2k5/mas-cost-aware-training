import os
import json
from mas_demo.contracts import DiagnosisReport
from mas_demo.llm.gateway import OpenRouterGateway, GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway


def diagnose(spec, evidence, history, gateway=None):
    gateway = gateway or create_gateway(spec)
    metadata = {}
    try:
        report, metadata = gateway.complete({"schema_version": 2, "evidence": evidence.model_dump(), "history": history,
                                              "allowed_interventions": [a.model_dump() for a in spec.intervention_allowlist],
                                              "task": f"{spec.dataset} classification / SmallCNN"})
        metadata["output"] = report.model_dump(exclude={"mode"})
        metadata["validation_inputs"] = {"available_epochs": evidence.epochs,
                                         "allowed_interventions": [a.model_dump() for a in spec.intervention_allowlist]}

        def reject(code, message, **details):
            metadata["validation_error"] = {"code": code, **details}
            raise GatewayError(message)

        for hypothesis in report.hypotheses:
            invalid_epochs = sorted(set(hypothesis.evidence_epochs) - set(evidence.epochs))
            if not hypothesis.evidence_epochs or invalid_epochs:
                reject("unavailable_epochs",
                       f"Hypothesis {hypothesis.hypothesis_id}: cited epochs {hypothesis.evidence_epochs}; available epochs {evidence.epochs}",
                       hypothesis_id=hypothesis.hypothesis_id, invalid_epochs=invalid_epochs)
            forbidden = [a.model_dump() for a in hypothesis.interventions if a not in spec.intervention_allowlist]
            if not hypothesis.interventions or forbidden:
                reject("forbidden_intervention",
                       f"Hypothesis {hypothesis.hypothesis_id}: empty or forbidden interventions {json.dumps(forbidden)}; see metadata.validation_inputs for allowed values",
                       hypothesis_id=hypothesis.hypothesis_id, forbidden_interventions=forbidden)
        if report.status == "DIAGNOSED" and not report.hypotheses:
            reject("missing_hypothesis", "No supported hypothesis")
        return report, metadata
    except GatewayError as error:
        return DiagnosisReport(status="ABSTAIN", mode=spec.analysis_mode, reason=str(error)), failure_metadata(error, "analysis", spec, metadata)

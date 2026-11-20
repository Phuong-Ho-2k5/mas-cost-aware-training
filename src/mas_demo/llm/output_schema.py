"""State-grounded generation constraints; business validation remains authoritative."""
from copy import deepcopy
import math


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _enum(values, kind="string"):
    return {"type": kind, "enum": list(dict.fromkeys(values))}


def _epochs(payload):
    return [n for n in payload.get("evidence", {}).get("epochs", []) if type(n) is int]


def _different_value(schema, current):
    """Exclude copied scalar values while retaining free choices and component switches."""
    if "anyOf" in schema:
        branches = [_different_value(branch, current) for branch in schema["anyOf"]
                    if not (branch.get("type") == "null" and current is None)]
        return {"anyOf": branches} if len(branches) != 1 else branches[0]
    if schema.get("type") not in ("number", "integer") or current is None:
        return deepcopy(schema)
    minimum = schema.get("minimum", -math.inf)
    maximum = schema.get("maximum", math.inf)
    lower = max(minimum, schema.get("exclusiveMinimum", -math.inf))
    upper = min(maximum, schema.get("exclusiveMaximum", math.inf))
    if schema["type"] == "integer":
        if math.isfinite(lower):
            lower = (math.floor(lower) + 1 if schema.get("exclusiveMinimum") == lower else math.ceil(lower))
        if math.isfinite(upper):
            upper = (math.ceil(upper) - 1 if schema.get("exclusiveMaximum") == upper else math.floor(upper))
    branches = []
    if lower < current:
        branches.append({**deepcopy(schema), "exclusiveMaximum": current})
    if upper > current:
        branches.append({**deepcopy(schema), "exclusiveMinimum": current})
    return {"anyOf": branches} if len(branches) != 1 else branches[0]


def _changes(patch, current):
    fields = patch["properties"]
    special = {"momentum", "beta1", "beta2", "epsilon", "scheduler_step_size", "scheduler_gamma", "scheduler_t_max"}
    common = set(fields) - special
    branches = []
    for optimizer in fields["optimizer"]["enum"]:
        for scheduler in fields["scheduler"]["enum"]:
            active = {"momentum"} if optimizer == "sgd" else {"beta1", "beta2", "epsilon"}
            if scheduler == "step":active |= {"scheduler_step_size", "scheduler_gamma"}
            if scheduler == "cosine":active.add("scheduler_t_max")
            properties = {name: deepcopy(value) for name, value in fields.items() if name in common | active}
            properties["optimizer"] = _enum([optimizer])
            properties["scheduler"] = _enum([scheduler])
            required = []
            if optimizer != current["optimizer"]:required.append("optimizer")
            else:properties.pop("optimizer")
            if scheduler != current["scheduler"]:required.append("scheduler")
            else:properties.pop("scheduler")
            for name in list(properties):
                if name not in ("optimizer", "scheduler"):
                    properties[name] = _different_value(properties[name], current.get(name))
            branches.append({"type": "object", "properties": properties, "required": required,
                             "additionalProperties": False, "minProperties": 1})
    return {"anyOf": branches}


def grounded_schema(role, schema, payload):
    schema = deepcopy(schema)
    definitions = schema.get("$defs", {})
    properties = schema["properties"]
    reason = properties["reason"]
    if role == "analysis_detection" and payload.get("input_format") == "epoch_logs":
        epochs = _epochs(payload)
        refs = {"type": "array", "items": _enum(epochs, "integer") if epochs else {"type": "integer"}, "maxItems": min(8, len(epochs))}
        branches = [_object({"status": _enum(["ABSTAIN"]), "evidence_epochs": refs, "reason": reason})]
        measured = {e.get("epoch") for e in payload.get("evidence", {}).get("recent_events", [])}
        if measured & set(epochs):
            branches.append(_object({"status": _enum(["DIAGNOSED"]), "evidence_epochs": {**refs, "minItems": 1, "items": _enum(sorted(measured & set(epochs)), "integer")}, "reason": reason}))
    elif role == "analysis_detection":
        from mas_demo.observation.facts import fact_has_measurement
        ids = [f["fact_id"] for f in payload.get("facts", [])]
        observed_ids = [f["fact_id"] for f in payload.get("facts", []) if fact_has_measurement(f)]
        references = {"type": "array", "items": _enum(ids) if ids else {"type": "string"}, "maxItems": 4 if ids else 0}
        branches = [_object({"status": _enum(["ABSTAIN"]), "observation_ids": references, "reason": reason})]
        if observed_ids:
            branches.append(_object({"status": _enum(["DIAGNOSED"]), "observation_ids": {**references, "items": _enum(observed_ids), "minItems": 1}, "reason": reason}))
    elif role == "analysis_hypotheses":
        ids = [f["fact_id"] for f in payload.get("facts", [])]
        raw_logs = payload.get("input_format") == "epoch_logs"
        hypothesis = definitions["EpochLogHypothesis" if raw_logs else "GeneratedHypothesis"]["properties"]
        if not raw_logs:
            hypothesis["observation_ids"] = {"type": "array", "items": _enum(ids), "minItems": 1, "maxItems": 4}
        allowed_epochs = _epochs(payload)
        if raw_logs:
            observed_epochs = {e.get("epoch") for e in payload.get("evidence", {}).get("recent_events", [])}
            allowed_epochs = [epoch for epoch in allowed_epochs if epoch in observed_epochs]
        hypothesis["evidence_epochs"] = {"type": "array", "items": _enum(allowed_epochs, "integer"), "minItems": 1, "maxItems": 8}
        claims = []
        config_facts = payload.get("config_facts", [])
        if raw_logs and payload.get("current_config"):
            from mas_demo.contracts import TrainingConfig
            from mas_demo.observation.facts import build_config_facts
            config_facts = [f.model_dump() for f in build_config_facts(TrainingConfig.model_validate(payload["current_config"]))]
        for fact in config_facts:
            value = fact["value"]
            kind = "null" if value is None else "string" if isinstance(value, str) else "integer" if type(value) is int else "number"
            claims.append(_object({"field": _enum([fact["field"]]),
                "observed_value": {"type": kind, "enum": [value]},
                "claim": _enum(["setting", "active_effect" if fact["active"] else "missing_effect"])}))
        if claims:
            definitions["ConfigClaim"] = {"anyOf": claims}
        else:
            hypothesis["config_claims"] = {"type": "array", "maxItems": 0}
        branches = [_object({"hypotheses": properties["hypotheses"], "reason": reason})]
    elif role == "analysis":
        empty = _object({"status": _enum(["ABSTAIN"]),
                         "hypotheses": {"type": "array", "maxItems": 0}, "reason": reason})
        branches = [empty]
        epochs = _epochs(payload)
        if epochs:
            definitions["AnalysisHypothesis"]["properties"]["evidence_epochs"] = {
                "type": "array", "items": _enum(epochs, "integer"), "minItems": 1, "maxItems": 8}
            hypotheses = {**properties["hypotheses"], "minItems": 3}
            branches.append(_object({"status": _enum(["DIAGNOSED"]), "hypotheses": hypotheses, "reason": reason}))
    elif role == "planner":
        definitions["ConfigPatch"]["properties"]["reason"]["maxLength"] = 120
        definitions["ConfigProposal"]["properties"]["expected_effect"]["maxLength"] = 120
        empty = _object({"proposals": {"type": "array", "maxItems": 0},
                         "verification_epochs": {"type": "null"}, "reason": reason})
        branches = [empty]
        epochs = _epochs(payload)
        identifiers = [h["hypothesis_id"] for h in (payload.get("diagnosis") or {}).get("hypotheses", [])
                       if isinstance(h.get("hypothesis_id"), str)]
        required_count = len(identifiers)
        remaining = payload.get("remaining_epochs", 0)
        current = payload.get("current_config", {})
        base = payload.get("base_config_hash")
        if (epochs and identifiers and required_count == 3 and len(set(identifiers)) == required_count
            and (payload.get("diagnosis") or {}).get("status", "DIAGNOSED") == "DIAGNOSED"
            and type(remaining) is int and remaining > 0 and base and current):
            definitions["ConfigProposal"]["properties"]["hypothesis_id"] = _enum(identifiers)
            definitions["ConfigPatch"]["properties"]["base_config_hash"] = _enum([base])
            definitions["ConfigPatch"]["properties"]["evidence_epochs"] = {
                "type": "array", "items": _enum(epochs, "integer"), "minItems": 1, "maxItems": 8}
            definitions["TrainingPatch"] = _changes(definitions["TrainingPatch"], current)
            proposals = {**properties["proposals"], "minItems": required_count, "maxItems": required_count}
            branches = [_object({"proposals": proposals,
                "verification_epochs": {"type": "integer", "minimum": 1, "maximum": remaining}, "reason": reason})]
    elif role == "evaluation":
        comparable = payload.get("control") is not None and any(v is not None for v in payload.get("trials", {}).values())
        decisions = ["KEEP_MAIN", "INCONCLUSIVE"] if comparable else ["INCONCLUSIVE"]
        branches = [_object({"decision": _enum(decisions), "selected_branch": _enum(["main"]), "reason": reason})]
        eligible = payload.get("eligible_branches", [])
        if comparable and eligible:
            branches.append(_object({"decision": _enum(["PROMOTE"]), "selected_branch": _enum(eligible), "reason": reason}))
    else:
        return schema
    result = {"anyOf": branches}
    if definitions:result["$defs"] = definitions
    return result

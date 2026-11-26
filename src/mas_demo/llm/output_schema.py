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
    """Exclude copied or previously rejected scalar values while retaining valid choices."""
    if "anyOf" in schema:
        branches = []
        for branch in schema["anyOf"]:
            if branch.get("type") == "null" and current is None:
                continue
            updated = _different_value(branch, current)
            if "anyOf" in updated:
                branches.extend(updated["anyOf"])
            else:
                branches.append(updated)
        return {"anyOf": branches} if len(branches) != 1 else branches[0]
    if schema.get("type") == "boolean" and isinstance(current, bool):
        if "enum" in schema:
            choices = [v for v in schema["enum"] if v != current]
            return {**deepcopy(schema), "enum": choices} if choices else deepcopy(schema)
        return {"type": "boolean", "enum": [not current]}
    if "enum" in schema and current is not None:
        choices = [v for v in schema["enum"] if v != current]
        if choices:
            return {**deepcopy(schema), "enum": choices}
        return deepcopy(schema)
    if schema.get("type") not in ("number", "integer") or current is None or isinstance(current, bool):
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
    if current < lower or (current == lower and schema.get("exclusiveMinimum") == lower):
        return deepcopy(schema)
    if current > upper or (current == upper and schema.get("exclusiveMaximum") == upper):
        return deepcopy(schema)
    branches = []
    if lower < current:
        low_branch = deepcopy(schema)
        low_branch.pop("maximum", None)
        low_branch["exclusiveMaximum"] = current
        branches.append(low_branch)
    if upper > current:
        high_branch = deepcopy(schema)
        high_branch.pop("minimum", None)
        high_branch["exclusiveMinimum"] = current
        branches.append(high_branch)
    if not branches:
        return deepcopy(schema)
    return {"anyOf": branches} if len(branches) != 1 else branches[0]


def _rejected_single_values(payload, base):
    rejected = {}
    for row in payload.get("history") or []:
        if row.get("base_config_hash") != base or row.get("status") != "REJECT":
            continue
        diff = row.get("diff") or {}
        if len(diff) == 1:
            field, pair = next(iter(diff.items()))
            if isinstance(pair, dict) and "after" in pair:
                vals = rejected.setdefault(field, [])
                if pair["after"] not in vals:
                    vals.append(pair["after"])
    for err in payload.get("validation_errors") or []:
        if err.get("code") != "ALREADY_TRIED":
            continue
        diff = err.get("diff") or {}
        if len(diff) == 1:
            field, pair = next(iter(diff.items()))
            if isinstance(pair, dict) and "after" in pair:
                vals = rejected.setdefault(field, [])
                if pair["after"] not in vals:
                    vals.append(pair["after"])
    return rejected


_NUMERIC_CAPS = {
    "learning_rate": 0.5,
    "weight_decay": 0.1,
    "grad_clip_norm": 100.0,
    "grad_clip_value": 100.0,
    "batch_size": 1024,
    "gradient_accumulation": 16,
    "epsilon": 0.01,
    "scheduler_step_size": 50,
    "scheduler_gamma": 0.99,
    "scheduler_t_max": 100,
    "scheduler_patience": 20,
    "early_stop_patience": 50,
}


def _cap_numeric(schema, cap):
    schema = deepcopy(schema)
    if "anyOf" in schema:
        schema["anyOf"] = [_cap_numeric(branch, cap) for branch in schema["anyOf"]]
        return schema
    if schema.get("type") in ("number", "integer") and "maximum" not in schema and "exclusiveMaximum" not in schema:
        schema["maximum"] = cap
    return schema


def _changes(patch, current, rejected=None):
    fields = {
        name: _cap_numeric(
            spec,
            max(cap, current[name] * 2) if isinstance(current.get(name), (int, float)) else cap,
        ) if (cap := _NUMERIC_CAPS.get(name)) is not None else spec
        for name, spec in patch["properties"].items()
    }
    special = {
        "momentum",
        "beta1",
        "beta2",
        "epsilon",
        "scheduler_step_size",
        "scheduler_gamma",
        "scheduler_t_max",
        "scheduler_patience",
    }
    common = set(fields) - special
    branches = []
    for optimizer in fields["optimizer"]["enum"]:
        for scheduler in fields["scheduler"]["enum"]:
            if optimizer == "sgd":
                active = {"momentum"}
            elif optimizer == "rmsprop":
                active = {"momentum", "epsilon"}
            else:
                active = {"beta1", "beta2", "epsilon"}
            if scheduler == "step":
                active |= {"scheduler_step_size", "scheduler_gamma"}
            elif scheduler == "cosine":
                active.add("scheduler_t_max")
            elif scheduler == "plateau":
                active |= {"scheduler_gamma", "scheduler_patience"}
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
                    for bad_val in (rejected or {}).get(name, [])[:3]:
                        properties[name] = _different_value(properties[name], bad_val)
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
        ws = payload.get("window_summary") or {}
        must_diagnose = bool(ws.get("numerical_error") or ws.get("stalled_window") or ws.get("unstable_window"))
        measured = {e.get("epoch") for e in payload.get("evidence", {}).get("recent_events", [])}
        branches = []
        if not (must_diagnose and (measured & set(epochs))):
            branches.append(_object({"evidence_epochs": refs, "reason": reason, "status": _enum(["ABSTAIN"])}))
        if measured & set(epochs):
            branches.append(_object({"evidence_epochs": {**refs, "minItems": 1, "items": _enum(sorted(measured & set(epochs)), "integer")}, "reason": reason, "status": _enum(["DIAGNOSED"])}))
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
            kind = "null" if value is None else "boolean" if isinstance(value, bool) else "string" if isinstance(value, str) else "integer" if type(value) is int else "number"
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
            hypotheses = {**properties["hypotheses"], "minItems": 1, "maxItems": 3}
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
        if (epochs and identifiers and 1 <= required_count <= 3 and len(set(identifiers)) == required_count
            and (payload.get("diagnosis") or {}).get("status", "DIAGNOSED") == "DIAGNOSED"
            and type(remaining) is int and remaining > 0 and base and current):
            definitions["ConfigProposal"]["properties"]["hypothesis_id"] = _enum(identifiers)
            definitions["ConfigPatch"]["properties"]["base_config_hash"] = _enum([base])
            definitions["ConfigPatch"]["properties"]["evidence_epochs"] = {
                "type": "array", "items": _enum(epochs, "integer"), "minItems": 1, "maxItems": 8}
            definitions["TrainingPatch"] = _changes(
                definitions["TrainingPatch"], current, rejected=_rejected_single_values(payload, base)
            )
            proposals = {**properties["proposals"], "minItems": required_count, "maxItems": required_count}
            max_vep = min(remaining, payload.get("max_verification_epochs", 4))
            branches = [_object({"proposals": proposals,
                "verification_epochs": {"type": "integer", "minimum": 1, "maximum": max(1, max_vep)}, "reason": reason})]
    elif role == "evaluation":
        control_exploded = payload.get("control") is None and payload.get("control_status") == "NUMERICAL_ERROR"
        has_trial = any(v is not None for v in payload.get("trials", {}).values())
        comparable = (payload.get("control") is not None or control_exploded) and has_trial
        decisions = ["KEEP_MAIN", "INCONCLUSIVE"] if (payload.get("control") is not None and has_trial) else ["INCONCLUSIVE"]
        eligible = payload.get("eligible_branches", [])
        branches = []
        if not (control_exploded and eligible):
            branches.append(_object({"decision": _enum(decisions), "selected_branch": _enum(["main"]), "reason": reason}))
        if comparable and eligible:
            branches.append(_object({"decision": _enum(["PROMOTE"]), "selected_branch": _enum(eligible), "reason": reason}))
    else:
        return schema
    result = {"anyOf": branches}
    if definitions:result["$defs"] = definitions
    return result

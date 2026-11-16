"""State-grounded generation constraints; business validation remains authoritative."""
from copy import deepcopy


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _enum(values, kind="string"):
    return {"type": kind, "enum": list(dict.fromkeys(values))}


def _epochs(payload):
    return [n for n in payload.get("evidence", {}).get("epochs", []) if type(n) is int]


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
            if scheduler != current["scheduler"]:required.append("scheduler")
            branches.append({"type": "object", "properties": properties, "required": required,
                             "additionalProperties": False, "minProperties": 1})
    return {"anyOf": branches}


def grounded_schema(role, schema, payload):
    schema = deepcopy(schema)
    definitions = schema.get("$defs", {})
    properties = schema["properties"]
    reason = properties["reason"]
    if role == "analysis":
        empty = _object({"status": _enum(["ABSTAIN"]),
                         "hypotheses": {"type": "array", "maxItems": 0}, "reason": reason})
        branches = [empty]
        epochs = _epochs(payload)
        if epochs:
            definitions["AnalysisHypothesis"]["properties"]["evidence_epochs"] = {
                "type": "array", "items": _enum(epochs, "integer"), "minItems": 1, "maxItems": 8}
            hypotheses = {**properties["hypotheses"], "minItems": 1}
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
        count = payload.get("max_trials", properties["proposals"].get("maxItems", 3))
        remaining = payload.get("remaining_epochs", 0)
        current = payload.get("current_config", {})
        base = payload.get("base_config_hash")
        if epochs and identifiers and type(count) is int and count > 0 and type(remaining) is int and remaining > 0 and base and current:
            definitions["ConfigProposal"]["properties"]["hypothesis_id"] = _enum(identifiers)
            definitions["ConfigPatch"]["properties"]["base_config_hash"] = _enum([base])
            definitions["ConfigPatch"]["properties"]["evidence_epochs"] = {
                "type": "array", "items": _enum(epochs, "integer"), "minItems": 1, "maxItems": 8}
            definitions["TrainingPatch"] = _changes(definitions["TrainingPatch"], current)
            proposals = {**properties["proposals"], "minItems": 1, "maxItems": count}
            branches.append(_object({"proposals": proposals,
                "verification_epochs": {"type": "integer", "minimum": 1, "maximum": remaining}, "reason": reason}))
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

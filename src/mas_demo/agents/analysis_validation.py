"""Check references and explicit claims, without claiming causal verification."""
from collections import defaultdict
import re
import unicodedata
from mas_demo.observation.facts import fact_has_measurement
from mas_demo.agents.config_grounding import explicit_config_references


def validate_epoch_detection(response, evidence):
    refs = response.evidence_epochs
    observed = {e.epoch for e in evidence.recent_events} & set(evidence.epochs)
    if len(refs) != len(set(refs)) or not set(refs) <= observed:
        return [dict(code="unavailable_epochs", reason="Viện dẫn epoch khác nhau và có trong logs")]
    if response.status == "DIAGNOSED":
        if not refs:
            return [dict(code="missing_epoch_records", reason="DIAGNOSED cần viện dẫn bản ghi worker có thật")]
    return []


def _normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def select_distinct_hypotheses(response):
    """Keep the highest-priority copy; preserve raw output in the caller's audit.

    Matching normalized cause/mechanism text catches repetition, not arbitrary
    semantic paraphrases. Never turn an invalid retained claim into a valid one.
    """
    kept, dropped = [], []
    for hypothesis in sorted(response.hypotheses, key=lambda h: h.rank):
        duplicate = next((old for old in kept if any(
            _normalized(getattr(hypothesis, field))
            and _normalized(getattr(hypothesis, field)) == _normalized(getattr(old, field))
            for field in ("cause", "mechanism"))), None)
        if duplicate is not None:
            dropped.append(dict(hypothesis_id=hypothesis.hypothesis_id,
                                duplicate_of=duplicate.hypothesis_id, rank=hypothesis.rank))
        else:
            kept.append(hypothesis)
    rank_changes = []
    # Equal ranks on genuinely different retained causes are still an error.
    if len({h.rank for h in kept}) == len(kept):
        for rank, hypothesis in enumerate(kept, 1):
            if hypothesis.rank != rank:
                rank_changes.append(dict(hypothesis_id=hypothesis.hypothesis_id,
                                         before=hypothesis.rank, after=rank))
        kept = [h.model_copy(update={"rank": rank}) for rank, h in enumerate(kept, 1)]
    return response.model_copy(update={"hypotheses": kept}), dict(
        input_count=len(response.hypotheses), kept_count=len(kept),
        dropped=dropped, rank_changes=rank_changes)


def validate_detection(response, facts):
    ids = set(f.fact_id for f in facts)
    if response.status == "DIAGNOSED" and not response.observation_ids:
        return [dict(code="missing_observations", reason="DIAGNOSED cần viện dẫn quan sát có thật")]
    if len(set(response.observation_ids)) != len(response.observation_ids):
        return [dict(code="duplicate_observation", reason="Chỉ viện dẫn mỗi quan sát một lần")]
    missing = set(response.observation_ids) - ids
    if missing:
        return [dict(code="unknown_observation", reason="Chỉ viện dẫn quan sát được cung cấp", fact_ids=sorted(missing))]
    if response.status == "DIAGNOSED":
        observed = {f.fact_id for f in facts if fact_has_measurement(f)}
        if not set(response.observation_ids) <= observed:
            return [dict(code="missing_measurements", reason="Pattern thiếu dữ liệu chưa phải quan sát bất thường; cần viện dẫn số đo có thật")]
    return []


def validate_hypotheses(response, evidence, facts, config_facts):
    errors = []
    hypotheses = response.hypotheses
    if not 1 <= len(hypotheses) <= 3:
        errors.append(dict(code="invalid_hypothesis_count", reason="Trả từ một đến ba hypothesis khác nhau"))
    ids = [h.hypothesis_id for h in hypotheses]
    if len(ids) != len(set(ids)):
        errors.append(dict(code="duplicate_hypothesis", reason="Dùng hypothesis_id khác nhau", hypothesis_ids=ids))
    if {h.rank for h in hypotheses} != set(range(1, len(hypotheses) + 1)):
        errors.append(dict(code="invalid_hypothesis_ranking", reason="Với N hypothesis, dùng mỗi rank 1 đến N đúng một lần", hypothesis_ids=ids))
    for field in ("cause", "mechanism"):
        groups = defaultdict(list)
        for h in hypotheses:
            groups[_normalized(getattr(h, field))].append(h.hypothesis_id)
        for text, group in groups.items():
            if not text or len(group) > 1:
                errors.append(dict(code=f"invalid_hypothesis_{field}s", reason=f"Các {field} phải khác nhau và không rỗng; chỉ đổi lời văn không làm cơ chế khác đi", hypothesis_ids=group))
    observations = {f.fact_id: f for f in facts}
    config = {f.field: f for f in config_facts}
    for h in hypotheses:
        def error(code, reason, **details):
            errors.append(dict(code=code, reason=reason, hypothesis_ids=[h.hypothesis_id], **details))
        available = set(evidence.epochs)
        if not hasattr(h, "observation_ids"):
            available &= {event.epoch for event in evidence.recent_events}
        if not h.evidence_epochs or not set(h.evidence_epochs) <= available:
            error("unavailable_epochs", "Chỉ viện dẫn epoch có trong evidence")
        references = getattr(h, "observation_ids", [])
        if len(references) != len(set(references)):
            error("duplicate_observation", "Chỉ viện dẫn mỗi fact một lần")
        for fact_id in references:
            fact = observations.get(fact_id)
            if fact is None:
                error("unknown_observation", "Chỉ dùng fact_id được cung cấp", fact_ids=[fact_id])
            elif not set(fact.epochs) <= set(h.evidence_epochs):
                error("incomplete_evidence_epochs", "Viện dẫn đủ epoch hỗ trợ fact này", fact_ids=[fact_id], required_epochs=fact.epochs)
        mentioned, absent, active = explicit_config_references(h.cause + ". " + h.mechanism)
        declared = {claim.field for claim in h.config_claims}
        missing = mentioned - declared
        if missing:
            error("undeclared_config_claim", "Khai báo config_claims cho mọi field cấu hình được nêu trong cause/mechanism, hoặc bỏ khẳng định đó",
                  fields=sorted(missing), actual_config=[config[f].model_dump() for f in sorted(missing) if f in config])
        contradicted = sorted({f for f in absent if f in config and config[f].active}
                              | {f for f in active if f in config and not config[f].active})
        if contradicted:
            error("contradictory_config_prose", "Cause/mechanism nói sai trạng thái bật/tắt thực tế của thành phần; sửa nội dung, không chỉ đổi claim",
                  fields=contradicted, actual_config=[config[f].model_dump() for f in contradicted])
        if len(declared) != len(h.config_claims):
            error("duplicate_config_claim", "Mỗi field chỉ khai báo một config_claim")
        for claim in h.config_claims:
            fact = config.get(claim.field)
            if fact is None:
                error("unknown_config_field", "Chỉ dùng field cấu hình được cung cấp", fields=[claim.field])
            elif claim.observed_value != fact.value:
                error("incorrect_config_value", "Dùng đúng giá trị cấu hình đang có", fields=[claim.field], actual_value=fact.value)
            elif claim.claim == "active_effect" and not fact.active:
                error("inactive_config_effect", "Field này không hoạt động; không quy tác động đang diễn ra cho nó", fields=[claim.field])
            elif claim.claim == "missing_effect" and fact.active:
                error("active_config_effect", "Field này đang hoạt động; không nói tác động của nó bị thiếu", fields=[claim.field])
    return errors

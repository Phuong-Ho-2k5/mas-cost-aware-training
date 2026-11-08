import json

from mas_demo.contracts import TaskSpec


def test_fixture_covers_thirty_independent_cases():
    from mas_demo.llm.benchmark import load_cases
    cases = load_cases()
    assert len(cases) == len({c["id"] for c in cases}) == 30
    assert {role: sum(c["role"] == role for c in cases) for role in ("analysis", "planner", "evaluation")} == {"analysis": 10, "planner": 10, "evaluation": 10}


def test_benchmark_measures_raw_invalid_output_as_failure(provider):
    from mas_demo.llm.benchmark import run_case
    gateway, state = provider
    state["responses"]["training_diagnosis"] = {"not_a_diagnosis": True}
    record = run_case({"id": "missing", "role": "analysis", "scenario": "insufficient", "expected": "ABSTAIN"}, TaskSpec(), gateway)
    assert record["schema_valid"] is False
    assert record["correct"] is False  # Safe fallback must not score as model success.
    assert record["raw_output"] == '{"not_a_diagnosis": true}'


def test_benchmark_measures_valid_but_forbidden_actions(provider):
    from mas_demo.llm.benchmark import run_case
    gateway, state = provider
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [{
        "hypothesis_id": "wrong", "cause": "Loss unstable", "evidence_epochs": [99],
        "interventions": [{"parameter": "learning_rate", "operation": "multiply", "value": 0.3}],
        "confidence": 0.5}], "reason": "Try this"}
    record = run_case({"id": "wrong", "role": "analysis", "scenario": "unstable", "expected": "DIAGNOSED"}, TaskSpec(), gateway)
    assert record["schema_valid"] is True
    assert record["semantic_valid"] is False and record["correct"] is False


def test_report_does_not_hide_role_failures_or_transport_errors():
    from mas_demo.llm.benchmark import summarize
    rows = [dict(role="analysis", correct=True, schema_valid=True, semantic_valid=True, latency_seconds=1, transport_ok=True),
            dict(role="evaluation", correct=False, schema_valid=False, semantic_valid=False, latency_seconds=3, transport_ok=False)]
    report = summarize(rows)
    assert report["count"] == 2 and report["schema_valid_rate"] == 0.5
    assert report["roles"]["evaluation"]["correct_rate"] == 0
    assert report["roles"]["planner"]["count"] == 0
    assert report["demo_thresholds_met"] is False


def test_json_schema_success_does_not_hide_cross_field_validation_failure(provider):
    from mas_demo.llm.benchmark import run_case
    gateway, state = provider
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [{
        "hypothesis_id": "wrong", "cause": "Loss unstable", "evidence_epochs": [3, 4],
        "interventions": [{"parameter": "momentum", "operation": "multiply", "value": 0.5}],
        "confidence": 0.5}], "reason": "Wrong operation"}
    record = run_case({"id": "cross-field", "role": "analysis", "scenario": "unstable", "expected": "DIAGNOSED"}, TaskSpec(), gateway)
    assert record["schema_valid"] is True
    assert record["semantic_valid"] is False and record["correct"] is False


def test_three_successful_probes_cannot_pass_full_benchmark_gate():
    from mas_demo.llm.benchmark import summarize
    rows = [dict(role=role, case_id=role, correct=True, schema_valid=True, semantic_valid=True,
                 latency_seconds=1, transport_ok=True) for role in ("analysis", "planner", "evaluation")]
    assert summarize(rows)["demo_thresholds_met"] is False

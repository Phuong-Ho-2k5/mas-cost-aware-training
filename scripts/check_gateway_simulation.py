"""Run against: postman simulation run postman/simulations/gateway.sim.yaml --port 4010 --no-history --output ndjson"""
import argparse
from mas_demo.llm.gateway import OpenRouterGateway, GatewayError
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.contracts import TaskSpec, EvidenceWindow, EpochEvent, Intervention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4010")
    args = parser.parse_args()
    result, _ = OpenRouterGateway("simulation-only", base_url=args.url+"/success").complete({})
    assert result.status == "ABSTAIN"
    gateway = OpenRouterGateway("simulation-only", model="simulation-model", base_url=args.url+"/success")
    spec = TaskSpec(analysis_mode="openrouter", dataset="synthetic")
    evidence = EvidenceWindow(epochs=[3, 4], branch_id="main", status="READY", signals=["plateau"])
    diagnosis, analysis_meta = diagnose(spec, evidence, [], gateway=gateway)
    assert diagnosis.status == "DIAGNOSED", diagnosis.reason
    trials, planner_meta = plan_with_agent(spec, diagnosis, [], 1, 2, gateway=gateway)
    assert [t.lr_factor for t in trials] == [0.1, 0.5], planner_meta
    for parameter, value in (("momentum", 0.5), ("weight_decay", 0.001), ("grad_clip_norm", 1.0)):
        intervention = Intervention(parameter=parameter, operation="set", value=value)
        scenario = TaskSpec(allowed_interventions=[intervention])
        report, _ = diagnose(scenario, evidence, [], gateway=gateway)
        options, meta = plan_with_agent(scenario, report, [], 1, 2, gateway=gateway)
        assert len(options) == 1 and options[0].intervention == intervention, meta
        print(parameter + ": generic diagnosis and plan verified (simulated)")
    def event(branch, accuracy, loss):
        return EpochEvent(epoch=6, branch_id=branch, train_loss=1, val_loss=loss,
                          val_accuracy=accuracy, learning_rate=0.1, grad_norm=1)
    verdict, evaluation_meta = evaluate_with_agent(spec, [event("main", 0.5, 1)],
        {"trial-2": [event("trial-2", 0.7, 0.8)]}, 6, gateway=gateway)
    assert verdict.decision == "PROMOTE" and verdict.selected_branch == "trial-2", evaluation_meta
    assert abs(verdict.gain - 0.2) < 1e-8
    for meta in (analysis_meta, planner_meta, evaluation_meta):
        assert meta["model"] == "simulation-model" and meta["usage"]["total_tokens"] == 15
        print(meta["agent"] + ": role output and HTTP metadata verified (simulated)")
    cases = [("rate-limit", "HTTP 429", 5), ("timeout", "transport", 0.1),
             ("success/malformed", "schema", 5)]
    for route, expected, timeout in cases:
        try:
            OpenRouterGateway("simulation-only", base_url=args.url+"/"+route,
                              timeout=timeout, retry_delay=0, deadline=2).complete({})
        except GatewayError as error:
            assert expected in str(error), str(error)
            print(route + ": fail-closed verified")
        else:
            raise AssertionError("Expected failure: " + route)
    print("Gateway simulation checks passed")


if __name__ == "__main__":
    main()

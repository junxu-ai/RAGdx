from __future__ import annotations

from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult


def main() -> None:
    result = EvaluationResult(
        retrieval={"context_precision": 0.63, "context_recall": 0.57, "context_entities_recall": 0.54},
        generation={"faithfulness": 0.79, "response_relevancy": 0.82, "noise_sensitivity": 0.31, "context_utilization": 0.61},
        e2e={"answer_correctness": 0.68, "citation_accuracy": 0.71},
        metadata={"dataset": "demo", "tools": ["ragas", "ragchecker"]},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report)
    print("Diagnosis summary:")
    print(report.summary)
    print("
Priority actions:")
    for item in report.priority_actions:
        print(f"- {item}")
    print("
Optimization plan:")
    for exp in plan.experiments:
        print(f"- {exp.name} [{exp.tool}] -> {exp.description}")


if __name__ == "__main__":
    main()

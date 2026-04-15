from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.schemas.models import EvaluationResult


def test_diagnosis_produces_hypotheses():
    result = EvaluationResult(
        retrieval={"context_precision": 0.60, "context_recall": 0.55},
        generation={"faithfulness": 0.78, "response_relevancy": 0.80, "noise_sensitivity": 0.30},
        e2e={"answer_correctness": 0.65},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    assert report.hypotheses
    assert report.optimization_candidates

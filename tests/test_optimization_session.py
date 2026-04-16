from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore


def test_optimize_session_simulate(tmp_path):
    result = EvaluationResult(
        retrieval={"context_precision": 0.60, "context_recall": 0.55},
        generation={"faithfulness": 0.78, "response_relevancy": 0.80, "noise_sensitivity": 0.30},
        e2e={"answer_correctness": 0.65, "citation_accuracy": 0.68},
        metadata={"dataset": "test"},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result, strategy="pareto_evolutionary", budget=6)
    session = OptimizationExecutor(root=tmp_path / ".ragdx").execute_plan(
        plan, baseline=result, strategy="pareto_evolutionary", mode="simulate"
    )
    assert session.completed_trials == session.total_trials
    assert session.trials
    assert any(t.config_path for t in session.trials)
    assert session.pareto_front_ids

    store = RunStore(tmp_path / ".ragdx")
    store.save_session(session)
    loaded = store.load_session(session.session_id)
    assert loaded.session_id == session.session_id

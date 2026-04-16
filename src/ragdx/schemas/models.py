from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


LayerName = Literal["retrieval", "generation", "e2e", "pipeline"]
Severity = Literal["low", "medium", "high", "critical"]
ToolName = Literal["ragas", "ragchecker", "dspy", "autorag", "langchain", "llamaindex", "manual"]
SearchStrategy = Literal["bayesian", "pareto_evolutionary"]
ExecutionMode = Literal["simulate", "prepare_only", "execute"]
TrialStatus = Literal["planned", "running", "done", "failed", "prepared"]
SessionStatus = Literal["planned", "running", "completed", "failed", "prepared"]
OptimizerStage = Literal["corpus", "retrieval", "generation", "orchestration", "joint"]
FeedbackKind = Literal["thumbs_up", "thumbs_down", "user_correction", "escalation", "hallucination", "latency", "cost", "policy"]


class DatasetRecord(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    answer: Optional[str] = None
    contexts: List[str] = Field(default_factory=list)
    reference_contexts: List[str] = Field(default_factory=list)
    citations: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    kind: Literal["query", "retrieve", "rerank", "pack", "generate", "verify", "tool", "judge"] = "query"
    name: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class QueryTrace(BaseModel):
    trace_id: str
    question: str
    answer: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    spans: List[TraceSpan] = Field(default_factory=list)
    token_usage: Dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    labels: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorScore(BaseModel):
    evaluator: str
    metric: str
    score: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackEvent(BaseModel):
    feedback_id: str
    query_id: Optional[str] = None
    kind: FeedbackKind
    severity: Severity = "medium"
    rating: Optional[float] = None
    note: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class EvaluatorCalibration(BaseModel):
    metric: str
    agreement_score: float = 0.0
    audit_sample_size: int = 0
    notes: str = ""


class CausalSignal(BaseModel):
    node: str
    component: LayerName
    posterior: float = 0.0
    prior: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    recommended_experiment: str = ""




class CausalEdge(BaseModel):
    source: str
    target: str
    weight: float = 0.0
    rationale: str = ""


class CausalGraph(BaseModel):
    nodes: List[CausalSignal] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    retrieval: Dict[str, float] = Field(default_factory=dict)
    generation: Dict[str, float] = Field(default_factory=dict)
    e2e: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    traces: List[QueryTrace] = Field(default_factory=list)
    evaluator_scores: List[EvaluatorScore] = Field(default_factory=list)
    feedback_events: List[FeedbackEvent] = Field(default_factory=list)
    calibrations: List[EvaluatorCalibration] = Field(default_factory=list)

    def score(self, metric: str, default: float | None = None) -> float | None:
        for bucket in (self.retrieval, self.generation, self.e2e):
            if metric in bucket:
                return bucket[metric]
        return default


class DiagnosisHypothesis(BaseModel):
    component: LayerName
    root_cause: str
    severity: Severity = "medium"
    confidence: float = 0.5
    evidence: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    summary: str
    expected_thresholds: Dict[str, float] = Field(default_factory=dict)
    metric_gaps: Dict[str, float] = Field(default_factory=dict)
    hypotheses: List[DiagnosisHypothesis] = Field(default_factory=list)
    optimization_candidates: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)
    causal_signals: List[CausalSignal] = Field(default_factory=list)
    causal_graph: CausalGraph = Field(default_factory=CausalGraph)
    evaluator_agreement: Dict[str, float] = Field(default_factory=dict)
    diagnosis_confidence: float = 0.0
    disambiguation_actions: List[str] = Field(default_factory=list)


class OptimizationExperiment(BaseModel):
    name: str
    tool: ToolName
    target_component: LayerName
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    objectives: Dict[str, float] = Field(default_factory=dict)
    search_space: Dict[str, List[Any]] = Field(default_factory=dict)
    search_strategy: SearchStrategy = "bayesian"
    max_trials: int = 8
    status: Literal["planned", "running", "done", "failed"] = "planned"
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    notes: str = ""
    config_artifacts: List[str] = Field(default_factory=list)
    stage: OptimizerStage = "joint"
    constraints: Dict[str, float] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)


class OptimizationPlan(BaseModel):
    objective_metric: str
    experiments: List[OptimizationExperiment] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)


class ToolRunResult(BaseModel):
    tool: ToolName
    success: bool
    payload: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class MetricComparison(BaseModel):
    metric: str
    current: float
    baseline: float
    delta: float
    direction: Literal["improved", "regressed", "unchanged"]


class OptimizationTrial(BaseModel):
    trial_id: str
    experiment_name: str
    tool: ToolName
    strategy: SearchStrategy
    status: TrialStatus = "planned"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    config_path: Optional[str] = None
    output_path: Optional[str] = None
    log_path: Optional[str] = None
    runner_command: Optional[str] = None
    return_code: Optional[int] = None
    objective_scores: Dict[str, float] = Field(default_factory=dict)
    utility: Optional[float] = None
    feasible: Optional[bool] = None
    constraint_violations: Dict[str, float] = Field(default_factory=dict)
    feasibility_penalty: float = 0.0
    pareto_dominance_count: int = 0
    pareto_front: bool = False
    logs: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: str = ""


class OptimizationSession(BaseModel):
    session_id: str
    created_at: str
    run_id: Optional[str] = None
    strategy: SearchStrategy
    mode: ExecutionMode = "simulate"
    status: SessionStatus = "planned"
    plan: OptimizationPlan
    total_trials: int = 0
    completed_trials: int = 0
    current_experiment: Optional[str] = None
    trials: List[OptimizationTrial] = Field(default_factory=list)
    best_trial_id: Optional[str] = None
    pareto_front_ids: List[str] = Field(default_factory=list)
    feasible_pareto_front_ids: List[str] = Field(default_factory=list)
    hypervolume: float = 0.0
    feasible_hypervolume: float = 0.0
    notes: str = ""


class SavedRun(BaseModel):
    run_id: str
    created_at: str
    name: str
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    baseline_run_id: Optional[str] = None
    latest_session_id: Optional[str] = None
    evaluation: EvaluationResult
    diagnosis: DiagnosisReport
    optimization_plan: OptimizationPlan

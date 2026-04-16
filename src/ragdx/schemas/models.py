from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


LayerName = Literal["retrieval", "generation", "e2e", "pipeline"]
Severity = Literal["low", "medium", "high", "critical"]
ToolName = Literal["ragas", "ragchecker", "dspy", "autorag", "manual"]
SearchStrategy = Literal["bayesian", "pareto_evolutionary"]
ExecutionMode = Literal["simulate", "prepare_only"]
TrialStatus = Literal["planned", "running", "done", "failed", "prepared"]
SessionStatus = Literal["planned", "running", "completed", "failed", "prepared"]


class DatasetRecord(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    answer: Optional[str] = None
    contexts: List[str] = Field(default_factory=list)
    reference_contexts: List[str] = Field(default_factory=list)
    citations: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    retrieval: Dict[str, float] = Field(default_factory=dict)
    generation: Dict[str, float] = Field(default_factory=dict)
    e2e: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_tool_outputs: Dict[str, Any] = Field(default_factory=dict)

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
    objective_scores: Dict[str, float] = Field(default_factory=dict)
    utility: Optional[float] = None
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

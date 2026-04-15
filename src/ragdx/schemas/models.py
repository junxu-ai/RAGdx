from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


LayerName = Literal["retrieval", "generation", "e2e", "pipeline"]
Severity = Literal["low", "medium", "high", "critical"]
ToolName = Literal["ragas", "ragchecker", "dspy", "autorag", "manual"]


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
    status: Literal["planned", "running", "done", "failed"] = "planned"
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    notes: str = ""


class OptimizationPlan(BaseModel):
    objective_metric: str
    experiments: List[OptimizationExperiment] = Field(default_factory=list)


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


class SavedRun(BaseModel):
    run_id: str
    created_at: str
    name: str
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    baseline_run_id: Optional[str] = None
    evaluation: EvaluationResult
    diagnosis: DiagnosisReport
    optimization_plan: OptimizationPlan

from __future__ import annotations

from ragdx.engines.root_cause import RuleBasedRootCauseAnalyzer
from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import DiagnosisReport, EvaluationResult


class RAGDiagnosisEngine:
    def __init__(self, analyzer: RuleBasedRootCauseAnalyzer | None = None, llm_explainer: LLMDiagnosisExplainer | None = None):
        self.analyzer = analyzer or RuleBasedRootCauseAnalyzer()
        self.llm_explainer = llm_explainer
        self.planner = OptimizationPlanner()

    def diagnose(self, result: EvaluationResult, use_llm: bool = False) -> DiagnosisReport:
        report = self.analyzer.analyze(result)
        if use_llm and self.llm_explainer is not None:
            return self.llm_explainer.explain(result, report)
        return report

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

    def diagnose(self, result: EvaluationResult, use_llm: bool = False, use_both: bool = False) -> DiagnosisReport:
        rule_report = self.analyzer.analyze(result)
        if use_both:
            if self.llm_explainer is None:
                raise ValueError("LLM diagnosis requested but no llm_explainer is configured.")
            llm_report = self.llm_explainer.explain(result, rule_report)
            return self.llm_explainer.summarize_both(result, rule_report, llm_report)
        if use_llm:
            if self.llm_explainer is None:
                raise ValueError("LLM diagnosis requested but no llm_explainer is configured.")
            return self.llm_explainer.explain(result, rule_report)
        return rule_report

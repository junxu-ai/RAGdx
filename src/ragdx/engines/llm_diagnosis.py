from __future__ import annotations

import json
from typing import Any, Callable, Dict

from ragdx.schemas.models import DiagnosisReport, EvaluationResult


DEFAULT_PROMPT = """
You are a senior RAG evaluation diagnostician.
Given thresholds, metrics, and an initial rule-based diagnosis, produce a refined diagnosis.
Return strict JSON with keys:
summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions.
Prefer causal explanations over generic recommendations.
""".strip()


class LLMDiagnosisExplainer:
    def __init__(self, llm_callable: Callable[[str], str | Dict[str, Any]], prompt_template: str = DEFAULT_PROMPT):
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template

    def explain(self, result: EvaluationResult, base_report: DiagnosisReport) -> DiagnosisReport:
        payload = {
            "thresholds": base_report.expected_thresholds,
            "metrics": {
                "retrieval": result.retrieval,
                "generation": result.generation,
                "e2e": result.e2e,
            },
            "initial_diagnosis": base_report.model_dump(),
            "metadata": result.metadata,
        }
        prompt = self.prompt_template + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
        raw = self.llm_callable(prompt)
        data = json.loads(raw) if isinstance(raw, str) else raw
        return DiagnosisReport(**data)

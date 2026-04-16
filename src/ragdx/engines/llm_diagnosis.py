from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict

from ragdx.schemas.models import DiagnosisReport, EvaluationResult


DEFAULT_REFINE_PROMPT = """
You are a senior RAG evaluation diagnostician.
Your task is to refine a deterministic diagnosis of a retrieval-augmented generation system.

You will receive:
1. target thresholds
2. measured metrics across retrieval, generation, and end-to-end layers
3. an initial rule-based diagnosis
4. metadata about the run

Reasoning requirements:
- Start from the metrics, not from stylistic preference.
- Distinguish retrieval recall failure, retrieval precision or ranking noise, generator grounding failure, citation failure, and broader pipeline failure.
- Separate primary bottlenecks from secondary symptoms.
- Prefer causal reasoning over rephrasing the rule-based diagnosis.
- Be conservative when evidence is weak or mixed.
- Convert the analysis into remediation actions in execution order.

Output requirements:
- Return strict JSON only.
- Use exactly these top-level keys:
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions
- summary must be concise and specific.
- hypotheses must be a list of objects with keys:
  component, root_cause, severity, confidence, evidence, recommended_actions
- confidence must be numeric in [0,1].
- severity must be one of: low, medium, high, critical.
- component must be one of: retrieval, generation, e2e, pipeline.
- expected_thresholds should preserve the input threshold mapping unless there is a clear reason to restate it.
- metric_gaps should preserve the gap mapping unless the input appears inconsistent.
- priority_actions should be ordered from highest to lowest leverage.
""".strip()


DEFAULT_SUMMARIZE_BOTH_PROMPT = """
You are a senior RAG diagnosis reviewer.
You are given two diagnosis reports for the same RAG evaluation result:
1. a deterministic rule-based diagnosis
2. an LLM-refined diagnosis

Your task is to synthesize them into one final diagnosis report.

Instructions:
- Preserve agreements between the two reports.
- Resolve conflicts carefully using the actual metrics and thresholds.
- Prefer the more evidence-backed explanation, not the more verbose one.
- Avoid duplicate hypotheses.
- Keep the final report practical for remediation planning.
- Priority actions should reflect the real execution order.
- Retrieval fixes should usually precede generator tuning when retrieval is the main bottleneck.

Output requirements:
- Return strict JSON only.
- Use exactly these top-level keys:
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions
- hypotheses must be deduplicated and ranked by likely leverage.
- summary should explicitly mention the dominant bottleneck and any secondary issue.
""".strip()


class LLMDiagnosisExplainer:
    def __init__(
        self,
        llm_callable: Callable[[str], str | Dict[str, Any]],
        prompt_template: str = DEFAULT_REFINE_PROMPT,
        summary_prompt_template: str = DEFAULT_SUMMARIZE_BOTH_PROMPT,
    ):
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template
        self.summary_prompt_template = summary_prompt_template

    @staticmethod
    def _coerce_json(raw: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _normalize_component(component: str) -> str:
        if not isinstance(component, str):
            return component
        normalized = component.strip().lower()
        if normalized in {"retrieval", "generation", "e2e", "pipeline"}:
            return normalized

        if "retrieval" in normalized:
            return "retrieval"
        if "generation" in normalized:
            return "generation"
        if "pipeline" in normalized or "observability" in normalized or "orchestr" in normalized:
            return "pipeline"
        if "e2e" in normalized or "end-to-end" in normalized or "end to end" in normalized or "citation" in normalized:
            return "e2e"

        parts = re.split(r"[\/,|]+", normalized)
        for part in parts:
            part = part.strip()
            if part in {"retrieval", "generation", "e2e", "pipeline"}:
                return part

        return normalized

    def _normalize_report_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data

        hypotheses = data.get("hypotheses")
        if isinstance(hypotheses, list):
            for hyp in hypotheses:
                if isinstance(hyp, dict) and "component" in hyp:
                    hyp["component"] = self._normalize_component(hyp["component"])
        return data

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
        prompt = f"{self.prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2)}"
        data = self._coerce_json(self.llm_callable(prompt))
        data = self._normalize_report_data(data)
        return DiagnosisReport(**data)

    def summarize_both(
        self,
        result: EvaluationResult,
        rule_report: DiagnosisReport,
        llm_report: DiagnosisReport,
    ) -> DiagnosisReport:
        payload = {
            "metrics": {
                "retrieval": result.retrieval,
                "generation": result.generation,
                "e2e": result.e2e,
            },
            "metadata": result.metadata,
            "rule_based_diagnosis": rule_report.model_dump(),
            "llm_diagnosis": llm_report.model_dump(),
        }
        prompt = f"{self.summary_prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2)}"
        data = self._coerce_json(self.llm_callable(prompt))
        data = self._normalize_report_data(data)
        return DiagnosisReport(**data)

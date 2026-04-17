"""
LLM-Powered Diagnosis and Explanation Engine

Main Idea:
This module provides LLM-powered diagnosis capabilities for RAG pipelines. It uses large language models to refine rule-based diagnoses, provide natural language explanations, and synthesize multiple diagnosis approaches.

Functionalities:
- Diagnosis refinement: Uses LLM to enhance rule-based diagnosis with causal reasoning
- Multi-diagnosis synthesis: Combines rule-based and LLM diagnoses into unified reports
- Structured output: Generates diagnosis reports with hypotheses, confidence scores, and action items
- Customizable prompts: Supports custom prompt templates for different diagnosis scenarios
- JSON output handling: Robust parsing of LLM responses into structured diagnosis reports

Key features:
- Causal analysis: Identifies root causes beyond surface-level metric analysis
- Evidence-based reasoning: Requires metrics and evidence for all conclusions
- Action prioritization: Orders remediation actions by expected leverage
- Confidence scoring: Provides confidence levels for all hypotheses and recommendations

Usage:
Initialize with an LLM callable function:

    def my_llm(prompt: str) -> str:
        # Your LLM implementation
        return response

    from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer

    explainer = LLMDiagnosisExplainer(llm_callable=my_llm)
    refined_report = explainer.explain(evaluation_result, rule_based_report)

For combined analysis:

    final_report = explainer.summarize_both(evaluation_result, rule_report, llm_report)

The LLM callable should accept a prompt string and return either a JSON string or dictionary.
"""

from __future__ import annotations

import json
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
5. optional evaluator agreement, traces, and feedback signals if available

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
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions, causal_signals, evaluator_agreement, diagnosis_confidence, disambiguation_actions
- summary must be concise and specific.
- hypotheses must be a list of objects with keys:
  component, root_cause, severity, confidence, evidence, recommended_actions
- causal_signals must be a list of objects with keys: node, component, posterior, prior, evidence, recommended_experiment
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
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions, causal_signals, evaluator_agreement, diagnosis_confidence, disambiguation_actions
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
            "traces": [t.model_dump() for t in result.traces[:10]],
            "feedback_events": [f.model_dump() for f in result.feedback_events[:20]],
            "evaluator_scores": [e.model_dump() for e in result.evaluator_scores[:50]],
            "calibrations": [c.model_dump() for c in result.calibrations],
        }
        prompt = f"{self.prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2)}"
        data = self._coerce_json(self.llm_callable(prompt))
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
            "traces": [t.model_dump() for t in result.traces[:10]],
            "feedback_events": [f.model_dump() for f in result.feedback_events[:20]],
            "evaluator_scores": [e.model_dump() for e in result.evaluator_scores[:50]],
            "calibrations": [c.model_dump() for c in result.calibrations],
            "rule_based_diagnosis": rule_report.model_dump(),
            "llm_diagnosis": llm_report.model_dump(),
        }
        prompt = f"{self.summary_prompt_template}\n\nINPUT:\n{json.dumps(payload, indent=2)}"
        data = self._coerce_json(self.llm_callable(prompt))
        return DiagnosisReport(**data)

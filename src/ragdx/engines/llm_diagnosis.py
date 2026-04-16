from __future__ import annotations

import json
import json_repair
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

Instructions:
- Identify the most plausible root causes for underperformance.
- Distinguish retrieval failures, ranking/noise failures, grounding failures, citation failures, and broader pipeline issues.
- Prefer causal reasoning over surface descriptions.
- Do not repeat the input diagnosis mechanically. Improve it where warranted.
- Keep recommendations concrete and prioritized.
- Only claim a bottleneck when the metrics support it.
- Be conservative when evidence is weak.

Output requirements:
- Return strict JSON only.
- Use exactly these top-level keys:
  summary, expected_thresholds, metric_gaps, hypotheses, optimization_candidates, priority_actions
- summary must be concise but specific.
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
- Priority actions should reflect the true execution order. Retrieval fixes should usually precede generator tuning when retrieval is the main bottleneck.

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
    def _find_json_end(text: str, start: int) -> int:
        brace_count = 0
        in_string = False
        escape = False
        i = start
        while i < len(text):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return i
            i += 1
        return -1

    @staticmethod
    def _coerce_json(raw: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start == -1:
                raise
            end = LLMDiagnosisEngine._find_json_end(text, start)
            if end == -1:
                # Fallback to old method
                end = text.rfind("}")
                if end == -1 or end <= start:
                    raise
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                try:
                    return json_repair.loads(text[start:end+1])
                except Exception:
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
        }
        prompt = self.prompt_template + "INPUT:" + json.dumps(payload, indent=2)
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
            "rule_based_diagnosis": rule_report.model_dump(),
            "llm_diagnosis": llm_report.model_dump(),
        }
        prompt = self.summary_prompt_template + "INPUT:" + json.dumps(payload, indent=2)
        data = self._coerce_json(self.llm_callable(prompt))
        return DiagnosisReport(**data)

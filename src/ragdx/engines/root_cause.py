from __future__ import annotations

from typing import Dict, List

from ragdx.core.thresholds import DEFAULT_THRESHOLDS, LOWER_IS_BETTER
from ragdx.schemas.models import DiagnosisHypothesis, DiagnosisReport, EvaluationResult


class RuleBasedRootCauseAnalyzer:
    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    def _gap(self, metric: str, value: float) -> float:
        target = self.thresholds.get(metric)
        if target is None:
            return 0.0
        if metric in LOWER_IS_BETTER:
            return round(max(0.0, value - target), 4)
        return round(max(0.0, target - value), 4)

    def _metric_gaps(self, result: EvaluationResult) -> Dict[str, float]:
        gaps: Dict[str, float] = {}
        for bucket in (result.retrieval, result.generation, result.e2e):
            for metric, value in bucket.items():
                gap = self._gap(metric, value)
                if gap > 0:
                    gaps[metric] = gap
        return dict(sorted(gaps.items(), key=lambda kv: kv[1], reverse=True))

    def analyze(self, result: EvaluationResult) -> DiagnosisReport:
        gaps = self._metric_gaps(result)
        cp = result.score("context_precision", 1.0) or 1.0
        cr = result.score("context_recall", 1.0) or 1.0
        cer = result.score("context_entities_recall", cr) or cr
        faith = result.score("faithfulness", 1.0) or 1.0
        rel = result.score("response_relevancy", 1.0) or 1.0
        util = result.score("context_utilization", 1.0) or 1.0
        noise = result.score("noise_sensitivity", 0.0) or 0.0
        hall = result.score("hallucination", 0.0) or 0.0
        ans = result.score("answer_correctness", result.score("answer_accuracy", 1.0) or 1.0) or 1.0
        cite = result.score("citation_accuracy", 1.0) or 1.0

        hypotheses: List[DiagnosisHypothesis] = []
        candidates: List[str] = []
        actions: List[str] = []

        if cr < self.thresholds["context_recall"] and cp >= self.thresholds["context_precision"]:
            hypotheses.append(DiagnosisHypothesis(
                component="retrieval",
                root_cause="evidence miss despite acceptable retrieval precision",
                severity="high",
                confidence=0.86,
                evidence=[
                    f"context_recall={cr:.2f} is below target {self.thresholds['context_recall']:.2f}",
                    f"context_precision={cp:.2f} is not the primary bottleneck",
                    f"entity recall proxy={cer:.2f} suggests missing supporting facts",
                ],
                recommended_actions=[
                    "Increase recall with hybrid retrieval or larger candidate pool before reranking.",
                    "Tune chunk size, overlap, and document segmentation.",
                    "Inspect query rewriting and metadata filters.",
                ],
            ))
            candidates.append("autorag_pipeline_search")
            actions.append("Prioritize retrieval recall experiments before generator prompt tuning.")

        if cp < self.thresholds["context_precision"]:
            hypotheses.append(DiagnosisHypothesis(
                component="retrieval",
                root_cause="retrieval noise or weak ranking quality",
                severity="high",
                confidence=0.84,
                evidence=[
                    f"context_precision={cp:.2f} is below target {self.thresholds['context_precision']:.2f}",
                    "Noisy contexts usually propagate to hallucination and citation failures.",
                ],
                recommended_actions=[
                    "Add or tune reranker.",
                    "Reduce top-k or separate recall stage from final evidence stage.",
                    "Use metadata or section-aware filters.",
                ],
            ))
            candidates.append("autorag_pipeline_search")
            actions.append("Improve ranking precision and evidence filtering.")

        if faith < self.thresholds["faithfulness"] and cr >= 0.7:
            hypotheses.append(DiagnosisHypothesis(
                component="generation",
                root_cause="generator is not grounding sufficiently on retrieved evidence",
                severity="high",
                confidence=0.83,
                evidence=[
                    f"faithfulness={faith:.2f} is below target {self.thresholds['faithfulness']:.2f}",
                    f"context_recall={cr:.2f} indicates at least some relevant evidence is available",
                    f"context_utilization={util:.2f} suggests evidence may not be used effectively",
                ],
                recommended_actions=[
                    "Optimize synthesis prompt to require evidence-backed answers.",
                    "Use structured answer templates with citation requirements.",
                    "Add answer compression or claim verification stage.",
                ],
            ))
            candidates.append("dspy_prompt_optimization")
            actions.append("Tune generator behavior after retrieval quality is acceptable.")

        if noise > self.thresholds["noise_sensitivity"] or hall > self.thresholds["hallucination"]:
            hypotheses.append(DiagnosisHypothesis(
                component="generation",
                root_cause="answer is fragile under distractors or unsupported reasoning",
                severity="high",
                confidence=0.81,
                evidence=[
                    f"noise_sensitivity={noise:.2f}",
                    f"hallucination={hall:.2f}",
                    "The generator likely overweights spurious or weakly relevant text.",
                ],
                recommended_actions=[
                    "Constrain the answer to quoted or cited evidence.",
                    "Use context packing with stronger ordering and section labels.",
                    "Add a verifier or claim-level grounding pass.",
                ],
            ))
            candidates.extend(["dspy_prompt_optimization", "joint_ablation_eval"])
            actions.append("Run ablations to separate noisy retrieval from generator overreach.")

        if ans < self.thresholds["answer_correctness"] and not hypotheses:
            hypotheses.append(DiagnosisHypothesis(
                component="pipeline",
                root_cause="end-to-end quality is weak but component-level signals are inconclusive",
                severity="medium",
                confidence=0.62,
                evidence=[f"answer_correctness={ans:.2f} is below target"],
                recommended_actions=[
                    "Run controlled ablations over retriever, reranker, and answer prompt.",
                    "Inspect difficult examples and metric-label alignment.",
                ],
            ))
            candidates.append("joint_ablation_eval")
            actions.append("Review evaluation set quality and run component ablations.")

        if cite < self.thresholds["citation_accuracy"]:
            hypotheses.append(DiagnosisHypothesis(
                component="e2e",
                root_cause="citation mapping is weaker than answer generation",
                severity="medium",
                confidence=0.72,
                evidence=[f"citation_accuracy={cite:.2f} is below target {self.thresholds['citation_accuracy']:.2f}"],
                recommended_actions=[
                    "Enforce sentence-level citation formatting.",
                    "Return passage ids with the answer synthesis step.",
                ],
            ))
            actions.append("Add explicit citation scaffolding in the generation prompt or response schema.")

        if not hypotheses:
            summary = "Metrics are close to the configured thresholds. No dominant bottleneck is detected."
        else:
            lead = hypotheses[0]
            summary = f"Primary bottleneck: {lead.root_cause}. {len(hypotheses)} diagnosis hypotheses were generated."

        return DiagnosisReport(
            summary=summary,
            expected_thresholds=self.thresholds,
            metric_gaps=gaps,
            hypotheses=hypotheses,
            optimization_candidates=sorted(set(candidates)),
            priority_actions=actions,
        )

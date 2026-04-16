from __future__ import annotations

from typing import Dict, List

from ragdx.schemas.models import DiagnosisReport, EvaluationResult, OptimizationExperiment, OptimizationPlan, SearchStrategy


class OptimizationPlanner:
    def _weights_for_component(self, component: str, objective_metric: str) -> Dict[str, float]:
        if component == "retrieval":
            return {"context_recall": 0.35, "context_precision": 0.25, objective_metric: 0.25, "citation_accuracy": 0.15}
        if component == "generation":
            return {"faithfulness": 0.35, "response_relevancy": 0.20, objective_metric: 0.25, "citation_accuracy": 0.20}
        return {objective_metric: 0.45, "citation_accuracy": 0.20, "faithfulness": 0.20, "context_recall": 0.15}

    def _retrieval_space(self) -> Dict[str, List[object]]:
        return {
            "chunk_size": [256, 384, 512, 768, 1024],
            "chunk_overlap": [32, 64, 96, 128],
            "retriever": ["bm25", "vector", "hybrid"],
            "reranker": ["none", "bge-reranker", "cross-encoder"],
            "top_k": [4, 6, 8, 10],
            "context_ordering": ["retrieval_score", "section_then_score", "diverse_then_score"],
        }

    def _generation_space(self) -> Dict[str, List[object]]:
        return {
            "optimizer": ["MIPROv2", "BootstrapFewShot", "GEPA"],
            "fewshot_count": [0, 2, 4, 6],
            "prompt_style": ["grounded_qa", "claim_then_evidence", "citation_first"],
            "temperature": [0.0, 0.2, 0.4],
            "max_citations": [1, 2, 3],
            "decomposition": [False, True],
        }

    def _joint_space(self) -> Dict[str, List[object]]:
        return {
            "retrieval_profile": ["recall_heavy", "balanced", "precision_heavy"],
            "generator_profile": ["grounded_qa", "citation_first"],
            "verifier": ["none", "claim_checker"],
            "context_budget": [3000, 4500, 6000],
        }

    def build_plan(
        self,
        report: DiagnosisReport,
        result: EvaluationResult | None = None,
        objective_metric: str = "answer_correctness",
        strategy: SearchStrategy = "bayesian",
        budget: int = 12,
    ) -> OptimizationPlan:
        experiments: List[OptimizationExperiment] = []
        rationale: List[str] = []
        candidates = set(report.optimization_candidates)

        if "autorag_pipeline_search" in candidates:
            exp_budget = max(4, budget // 2)
            experiments.append(
                OptimizationExperiment(
                    name="retrieval-pipeline-search",
                    tool="autorag",
                    target_component="retrieval",
                    description="Optimize retrieval, reranking, and context packing for better evidence quality.",
                    parameters={"diagnosis_summary": report.summary, "planner": "ragdx"},
                    objectives=self._weights_for_component("retrieval", objective_metric),
                    search_space=self._retrieval_space(),
                    search_strategy=strategy,
                    max_trials=exp_budget,
                    notes="AutoRAG-style search space generated from retrieval diagnosis.",
                )
            )
            rationale.append("Retrieval-focused search was selected because the diagnosis indicates evidence miss or ranking noise.")

        if "dspy_prompt_optimization" in candidates:
            exp_budget = max(4, budget // 2)
            experiments.append(
                OptimizationExperiment(
                    name="generator-prompt-optimization",
                    tool="dspy",
                    target_component="generation",
                    description="Optimize grounded answer synthesis and citation behavior.",
                    parameters={"diagnosis_summary": report.summary, "planner": "ragdx"},
                    objectives=self._weights_for_component("generation", objective_metric),
                    search_space=self._generation_space(),
                    search_strategy=strategy,
                    max_trials=exp_budget,
                    notes="DSPy optimization surface generated from generation diagnosis.",
                )
            )
            rationale.append("Generator-side optimization was selected because the diagnosis indicates grounding, citation, or response formation issues.")

        if "joint_ablation_eval" in candidates or not experiments:
            exp_budget = max(4, budget - sum(e.max_trials for e in experiments))
            experiments.append(
                OptimizationExperiment(
                    name="joint-pipeline-optimization",
                    tool="manual",
                    target_component="pipeline",
                    description="Run multi-objective search over retrieval/generation operating profiles when component-level root cause is mixed.",
                    parameters={"diagnosis_summary": report.summary, "planner": "ragdx"},
                    objectives=self._weights_for_component("pipeline", objective_metric),
                    search_space=self._joint_space(),
                    search_strategy="pareto_evolutionary" if strategy == "pareto_evolutionary" else strategy,
                    max_trials=exp_budget,
                    notes="Joint search is used when the pipeline bottleneck is mixed or inconclusive.",
                )
            )
            rationale.append("A joint search was added to resolve mixed retrieval and generation failures with direct end-to-end tradeoff tracking.")

        return OptimizationPlan(objective_metric=objective_metric, experiments=experiments, rationale=rationale)

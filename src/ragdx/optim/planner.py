from __future__ import annotations

from ragdx.schemas.models import DiagnosisReport, OptimizationExperiment, OptimizationPlan


class OptimizationPlanner:
    def build_plan(self, report: DiagnosisReport, objective_metric: str = "answer_correctness") -> OptimizationPlan:
        experiments = []
        if "autorag_pipeline_search" in report.optimization_candidates:
            experiments.append(
                OptimizationExperiment(
                    name="retrieval-pipeline-search",
                    tool="autorag",
                    target_component="retrieval",
                    description="Search chunking, retriever, reranker, context packing, and top-k settings.",
                    parameters={
                        "search_space": ["chunk_size", "overlap", "retriever", "reranker", "context_packing", "top_k"],
                        "objective": ["context_recall", "context_precision", objective_metric],
                        "priority": 1,
                    },
                )
            )
        if "dspy_prompt_optimization" in report.optimization_candidates:
            experiments.append(
                OptimizationExperiment(
                    name="generator-prompt-optimization",
                    tool="dspy",
                    target_component="generation",
                    description="Optimize answer synthesis, decomposition, and citation behavior with DSPy.",
                    parameters={
                        "optimizer_candidates": ["MIPROv2", "BootstrapRS", "GEPA"],
                        "target_metrics": ["faithfulness", "response_relevancy", objective_metric, "citation_accuracy"],
                        "priority": 2,
                    },
                )
            )
        if "joint_ablation_eval" in report.optimization_candidates:
            experiments.append(
                OptimizationExperiment(
                    name="joint-ablation",
                    tool="manual",
                    target_component="pipeline",
                    description="Run controlled ablations to separate retrieval, reranking, prompt, and citation-template effects.",
                    parameters={
                        "ablations": ["retrieval_only", "retrieval_plus_reranker", "prompt_only", "citation_prompt_only", "joint"],
                        "priority": 3,
                    },
                )
            )
        return OptimizationPlan(objective_metric=objective_metric, experiments=experiments)

"""
RAG Diagnosis Library (ragdx)

Main Idea:
This library provides comprehensive tools for diagnosing, evaluating, and optimizing Retrieval-Augmented Generation (RAG) pipelines. It enables users to identify performance bottlenecks, compare different RAG implementations, and automatically optimize configurations for better results.

Functionalities:
- Dataset Management: Handle and process RAG evaluation datasets
- Evaluation: Unified evaluation framework supporting multiple metrics and engines (RAGAS, RAGChecker, etc.)
- Diagnosis: Automated diagnosis of RAG pipeline issues with hypothesis generation and root cause analysis
- Optimization: Bayesian optimization and LLM-based planning for RAG parameter tuning
- Storage: Persistent storage for experiments, runs, and results
- UI/Dashboard: Web-based dashboard for visualization and monitoring
- CLI: Command-line interface for easy integration into workflows
- Adapters: Integration with popular RAG frameworks (LangChain, LlamaIndex, DSPy, AutoRAG)

Usage:
Import the main classes and use them in your RAG applications:

    from ragdx import RAGDiagnosisEngine, UnifiedEvaluator, OptimizationPlanner

    # Evaluate a RAG pipeline
    evaluator = UnifiedEvaluator()
    results = evaluator.evaluate(dataset, pipeline)

    # Diagnose issues
    diagnosis_engine = RAGDiagnosisEngine()
    report = diagnosis_engine.diagnose(results)

    # Optimize parameters
    planner = OptimizationPlanner()
    plan = planner.create_optimization_plan(report)
"""

from ragdx.schemas.models import (
    DatasetRecord,
    EvaluationResult,
    DiagnosisHypothesis,
    DiagnosisReport,
    OptimizationExperiment,
    OptimizationPlan,
    ToolRunResult,
    QueryTrace,
    TraceSpan,
    FeedbackEvent,
    CausalSignal,
)
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.optim.planner import OptimizationPlanner

__all__ = [
    "DatasetRecord",
    "EvaluationResult",
    "DiagnosisHypothesis",
    "DiagnosisReport",
    "OptimizationExperiment",
    "OptimizationPlan",
    "ToolRunResult",
    "QueryTrace",
    "TraceSpan",
    "FeedbackEvent",
    "CausalSignal",
    "RAGDiagnosisEngine",
    "UnifiedEvaluator",
    "OptimizationPlanner",
]

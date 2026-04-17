"""
Unified RAG Evaluation Framework

Main Idea:
This module provides a unified interface for evaluating RAG (Retrieval-Augmented Generation) pipelines using multiple evaluation engines. It integrates different evaluation tools and provides a consistent API for assessing RAG performance across various metrics.

Functionalities:
- Multi-engine evaluation: Supports RAGAS and RAGChecker evaluation engines
- Result merging: Combines evaluation results from different tools into a unified format
- Flexible configuration: Allows selective use of evaluation engines
- Raw score injection: Supports providing pre-computed scores for efficiency
- Comprehensive metrics: Covers retrieval, generation, and end-to-end evaluation aspects

Supported evaluation engines:
- RAGAS: Comprehensive RAG evaluation metrics
- RAGChecker: Specialized RAG evaluation with fact-checking capabilities

Usage:
Basic evaluation with all engines:

    from ragdx.core.evaluator import UnifiedEvaluator
    from ragdx.core.datasets import load_records

    evaluator = UnifiedEvaluator()
    records = load_records("dataset.jsonl")
    result = evaluator.evaluate(records)

Selective evaluation:

    result = evaluator.evaluate(records, use_ragas=True, use_ragchecker=False)

With pre-computed scores:

    ragas_scores = {"context_precision": 0.85, "faithfulness": 0.92}
    result = evaluator.evaluate(records, ragas_scores=ragas_scores)

The evaluation result contains metrics organized by category (retrieval, generation, e2e) and metadata about the evaluation process.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from ragdx.engines.ragas_adapter import RagasAdapter
from ragdx.engines.ragchecker_adapter import RAGCheckerAdapter
from ragdx.schemas.models import DatasetRecord, EvaluationResult


class UnifiedEvaluator:
    def __init__(self, ragas_adapter: Optional[RagasAdapter] = None, ragchecker_adapter: Optional[RAGCheckerAdapter] = None):
        self.ragas_adapter = ragas_adapter or RagasAdapter()
        self.ragchecker_adapter = ragchecker_adapter or RAGCheckerAdapter()

    def merge(self, *results: EvaluationResult) -> EvaluationResult:
        merged = EvaluationResult()
        for result in results:
            merged.retrieval.update(result.retrieval)
            merged.generation.update(result.generation)
            merged.e2e.update(result.e2e)
            merged.metadata.update(result.metadata)
            merged.raw_tool_outputs.update(result.raw_tool_outputs)
        return merged

    def evaluate(
        self,
        records: Iterable[DatasetRecord],
        ragas_scores: Mapping[str, float] | None = None,
        ragchecker_scores: Mapping[str, float] | None = None,
        use_ragas: bool = True,
        use_ragchecker: bool = True,
    ) -> EvaluationResult:
        outputs = []
        records = list(records)
        if use_ragas:
            outputs.append(self.ragas_adapter.evaluate(records, raw_scores=ragas_scores))
        if use_ragchecker:
            outputs.append(self.ragchecker_adapter.evaluate(records, raw_scores=ragchecker_scores))
        return self.merge(*outputs)

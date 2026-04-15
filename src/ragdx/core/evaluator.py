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

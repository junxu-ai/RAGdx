from __future__ import annotations

from typing import Any, Iterable, Mapping

from ragdx.core.normalization import RAGCHECKER_MAP
from ragdx.schemas.models import DatasetRecord, EvaluationResult


class RAGCheckerAdapter:
    def normalize_scores(self, raw_scores: Mapping[str, float]) -> EvaluationResult:
        result = EvaluationResult(metadata={"tool": "ragchecker"}, raw_tool_outputs={"ragchecker": dict(raw_scores)})
        for metric, value in raw_scores.items():
            mapped = RAGCHECKER_MAP.get(metric)
            if mapped:
                bucket, target = mapped
                getattr(result, bucket)[target] = float(value)
        return result

    def _to_ragchecker_records(self, records: Iterable[DatasetRecord]) -> list[dict[str, Any]]:
        rows = []
        for r in records:
            rows.append(
                {
                    "query": r.question,
                    "gt_answer": r.ground_truth or "",
                    "response": r.answer or "",
                    "retrieved_context": r.contexts,
                }
            )
        return rows

    def evaluate(self, records: Iterable[DatasetRecord], raw_scores: Mapping[str, float] | None = None, **kwargs: Any) -> EvaluationResult:
        records = list(records)
        try:
            import ragchecker  # noqa: F401
        except Exception as exc:
            raise ImportError("ragchecker is not installed. Install with `pip install ragdx[ragchecker]`.") from exc

        if raw_scores is not None:
            out = self.normalize_scores(raw_scores)
            out.metadata.update({"record_count": len(records), "mode": "precomputed"})
            return out

        return EvaluationResult(
            metadata={
                "tool": "ragchecker",
                "record_count": len(records),
                "mode": "prepared_only",
                "prepared_records": self._to_ragchecker_records(records),
                "note": "Prepared records in a ragchecker-friendly schema. Use this payload with your installed ragchecker workflow.",
            }
        )

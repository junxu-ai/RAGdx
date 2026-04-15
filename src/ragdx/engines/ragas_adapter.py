from __future__ import annotations

from typing import Any, Iterable, Mapping

from ragdx.core.normalization import RAGAS_MAP
from ragdx.schemas.models import DatasetRecord, EvaluationResult


class RagasAdapter:
    def normalize_scores(self, raw_scores: Mapping[str, float]) -> EvaluationResult:
        result = EvaluationResult(metadata={"tool": "ragas"}, raw_tool_outputs={"ragas": dict(raw_scores)})
        for metric, value in raw_scores.items():
            mapped = RAGAS_MAP.get(metric)
            if mapped:
                bucket, target = mapped
                getattr(result, bucket)[target] = float(value)
        return result

    def _to_ragas_records(self, records: Iterable[DatasetRecord]) -> list[dict[str, Any]]:
        rows = []
        for r in records:
            rows.append(
                {
                    "user_input": r.question,
                    "response": r.answer or "",
                    "retrieved_contexts": r.contexts,
                    "reference": r.ground_truth or "",
                    "reference_contexts": r.reference_contexts,
                }
            )
        return rows

    def evaluate(self, records: Iterable[DatasetRecord], raw_scores: Mapping[str, float] | None = None, **kwargs: Any) -> EvaluationResult:
        records = list(records)
        try:
            import ragas  # noqa: F401
        except Exception as exc:
            raise ImportError("ragas is not installed. Install with `pip install ragdx[ragas]`.") from exc

        if raw_scores is not None:
            out = self.normalize_scores(raw_scores)
            out.metadata.update({"record_count": len(records), "mode": "precomputed"})
            return out

        return EvaluationResult(
            metadata={
                "tool": "ragas",
                "record_count": len(records),
                "mode": "prepared_only",
                "prepared_records": self._to_ragas_records(records),
                "note": "Prepared records in a ragas-friendly schema. Use this payload with your installed ragas workflow or CLI.",
            }
        )

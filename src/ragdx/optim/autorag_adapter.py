from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import ToolRunResult


class AutoRAGAdapter:
    def build_search_spec(self, objective_metric: str = "answer_correctness") -> Dict[str, Any]:
        return {
            "framework": "autorag",
            "objective_metric": objective_metric,
            "search_dimensions": ["chunk_size", "overlap", "retriever", "reranker", "top_k", "context_ordering"],
            "recommended_split": {"train": 0.7, "dev": 0.15, "test": 0.15},
        }

    def run(self, **kwargs: Any) -> ToolRunResult:
        return ToolRunResult(
            tool="autorag",
            success=True,
            payload=self.build_search_spec(**kwargs),
            note="Planner stub. Use the generated search spec to build your AutoRAG YAML and execute the pipeline search in your environment.",
        )

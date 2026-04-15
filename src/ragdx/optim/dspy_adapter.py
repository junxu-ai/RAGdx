from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import ToolRunResult


class DSPyAdapter:
    def build_optimizer_spec(self, objective_metric: str = "answer_correctness", mode: str = "light") -> Dict[str, Any]:
        return {
            "framework": "dspy",
            "optimizers": ["MIPROv2", "BootstrapRS", "GEPA"],
            "objective_metric": objective_metric,
            "mode": mode,
            "expected_tunable_parts": ["instructions", "fewshot_demos", "citation_formatting", "decomposition"],
        }

    def run(self, **kwargs: Any) -> ToolRunResult:
        return ToolRunResult(
            tool="dspy",
            success=True,
            payload=self.build_optimizer_spec(**kwargs),
            note="Planner stub. Attach your DSPy program and metric function in your runtime environment.",
        )

from __future__ import annotations

from ragdx.schemas.models import EvaluationResult, MetricComparison


LOWER_IS_BETTER = {"noise_sensitivity", "hallucination"}


def compare_results(current: EvaluationResult, baseline: EvaluationResult) -> list[MetricComparison]:
    metrics = {}
    for bucket in (current.retrieval, current.generation, current.e2e):
        metrics.update(bucket)
    for bucket in (baseline.retrieval, baseline.generation, baseline.e2e):
        metrics.update(bucket)

    out: list[MetricComparison] = []
    for metric in sorted(metrics):
        c = current.score(metric)
        b = baseline.score(metric)
        if c is None or b is None:
            continue
        raw_delta = c - b
        better = raw_delta < 0 if metric in LOWER_IS_BETTER else raw_delta > 0
        direction = "unchanged" if abs(raw_delta) < 1e-9 else ("improved" if better else "regressed")
        out.append(MetricComparison(metric=metric, current=c, baseline=b, delta=round(raw_delta, 4), direction=direction))
    return out

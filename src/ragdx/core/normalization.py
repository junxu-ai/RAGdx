from __future__ import annotations

from typing import Dict, Tuple

RAGAS_MAP: Dict[str, Tuple[str, str]] = {
    "context_precision": ("retrieval", "context_precision"),
    "context_recall": ("retrieval", "context_recall"),
    "context_entity_recall": ("retrieval", "context_entities_recall"),
    "response_relevancy": ("generation", "response_relevancy"),
    "faithfulness": ("generation", "faithfulness"),
    "noise_sensitivity": ("generation", "noise_sensitivity"),
    "answer_correctness": ("e2e", "answer_correctness"),
    "answer_accuracy": ("e2e", "answer_accuracy"),
}

RAGCHECKER_MAP: Dict[str, Tuple[str, str]] = {
    "precision": ("retrieval", "context_precision"),
    "recall": ("retrieval", "context_recall"),
    "claim_recall": ("e2e", "answer_correctness"),
    "context_utilization": ("generation", "context_utilization"),
    "hallucination": ("generation", "hallucination"),
    "self_knowledge": ("generation", "noise_sensitivity"),
    "faithfulness": ("generation", "faithfulness"),
}

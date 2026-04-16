DEFAULT_THRESHOLDS = {
    "context_precision": 0.80,
    "context_recall": 0.80,
    "context_entities_recall": 0.75,
    "hit_rate_at_k": 0.85,
    "faithfulness": 0.90,
    "response_relevancy": 0.85,
    "context_utilization": 0.75,
    "noise_sensitivity": 0.20,
    "hallucination": 0.10,
    "answer_correctness": 0.85,
    "answer_accuracy": 0.85,
    "citation_accuracy": 0.85,
    "user_success_rate": 0.85,
}

LOWER_IS_BETTER = {"noise_sensitivity", "hallucination", "retrieval_latency_ms", "latency_ms", "cost_usd"}

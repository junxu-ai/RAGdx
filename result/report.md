# ragdx run report: demo-run

- Run ID: `7706d1ac71b7`
- Created at: 2026-04-15T23:41:17.484886+00:00
- Baseline run: `none`
- Tags: demo, baseline

## Summary
Primary bottleneck: retrieval noise or weak ranking quality. 3 diagnosis hypotheses were generated.

## Retrieval metrics
- context_precision: 0.6300
- context_recall: 0.5700
- context_entities_recall: 0.5400
- hit_rate_at_k: 0.6400

## Generation metrics
- faithfulness: 0.7900
- response_relevancy: 0.8200
- noise_sensitivity: 0.3100
- context_utilization: 0.6100
- hallucination: 0.1900

## End-to-end metrics
- answer_correctness: 0.6800
- citation_accuracy: 0.7100
- user_success_rate: 0.6900

## Hypotheses
- **retrieval noise or weak ranking quality** (retrieval, severity=high, confidence=0.84)
- **answer is fragile under distractors or unsupported reasoning** (generation, severity=high, confidence=0.81)
- **citation mapping is weaker than answer generation** (e2e, severity=medium, confidence=0.72)

## Planned experiments
- **retrieval-pipeline-search** via `autorag` targeting `retrieval`: Search chunking, retriever, reranker, context packing, and top-k settings.
- **generator-prompt-optimization** via `dspy` targeting `generation`: Optimize answer synthesis, decomposition, and citation behavior with DSPy.
- **joint-ablation** via `manual` targeting `pipeline`: Run controlled ablations to separate retrieval, reranking, prompt, and citation-template effects.
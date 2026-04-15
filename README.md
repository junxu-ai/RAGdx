# ragdx v3

A Python library for RAG evaluation, diagnosis, and optimization planning.

## What it does

- normalizes evaluation outputs from Ragas and RAGChecker into a unified schema
- diagnoses weak retrieval, generation, citation, and end-to-end behavior
- stages DSPy and AutoRAG optimization plans
- stores runs, compares baselines, and exports markdown reports
- provides a simple Streamlit dashboard

## Install

```bash
pip install -e .
```

## CLI

```bash
ragdx diagnose --eval-json examples/demo_evaluation.json
ragdx save --eval-json examples/demo_evaluation.json --name demo-run --tags demo,baseline
ragdx runs
ragdx compare --current-eval-json examples/demo_evaluation.json --baseline-eval-json examples/demo_evaluation_baseline.json
ragdx dashboard
```

## Run store

Saved runs are stored under `.ragdx/runs`.

## Notes

This package includes runnable diagnosis and planning logic. Live execution against Ragas, RAGChecker, DSPy, and AutoRAG still depends on the versions and runtime you install locally.

"""
LangChain Framework Adapter

Main Idea:
This module provides an adapter for optimizing RAG pipelines built with LangChain. It generates configuration specifications for LangChain-based retrieval chains and integrates with the optimization workflow.

Functionalities:
- Configuration generation: Creates runner specifications for LangChain experiments
- Parameter mapping: Maps optimization parameters to LangChain configuration
- Framework integration: Supports various LangChain components (LLMs, retrievers, vector stores)
- Contract definition: Defines the interface between optimization and execution

Supported LangChain components:
- LLM providers: OpenAI, Anthropic, local models
- Vector stores: FAISS, Chroma, Pinecone
- Retrievers: Similarity search, MMR, contextual compression
- Rerankers: Cross-encoder reranking, LLM-based reranking

Usage:
Build optimization spec:

    from ragdx.optim.langchain_adapter import LangChainAdapter

    adapter = LangChainAdapter()
    spec = adapter.build_runner_spec(experiment, parameters)

The adapter prepares configurations that can be executed by LangChain-based runners for optimization trials.
"""

from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import OptimizationExperiment, ToolRunResult


class LangChainAdapter:
    def build_runner_spec(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> Dict[str, Any]:
        llm_provider = parameters.get("llm_provider", "openai")
        return {
            "framework": "langchain",
            "entrypoint": "examples/run_langchain_trial.py",
            "objective_metric": experiment.parameters.get("objective_metric", "answer_correctness"),
            "objectives": experiment.objectives,
            "constraints": experiment.constraints,
            "program_contract": {
                "dataset_path": experiment.parameters.get("dataset_path", "examples/demo_dataset.jsonl"),
                "pipeline_module": experiment.parameters.get("pipeline_module", "examples.langchain_pipeline:create_pipeline"),
                "evaluator_mode": experiment.parameters.get("evaluator_mode", "offline"),
            },
            "runtime": {
                "provider": llm_provider,
                "vectorstore": parameters.get("vectorstore", "faiss"),
                "retriever_k": parameters.get("top_k", 6),
                "search_type": parameters.get("search_type", "similarity"),
                "reranker": parameters.get("reranker", "none"),
                "temperature": parameters.get("temperature", 0.0),
            },
            "search_parameters": parameters,
        }

    def run(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> ToolRunResult:
        return ToolRunResult(
            tool="langchain",
            success=True,
            payload=self.build_runner_spec(experiment, parameters),
            note="Config rendered for a LangChain retrieval chain runner. Set RAGDX_LANGCHAIN_RUNNER_CMD to execute it.",
        )

from __future__ import annotations

from typing import Any, Dict


def create_pipeline(config: Dict[str, Any]):
    try:
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain.chains.retrieval import create_retrieval_chain
    except Exception as exc:
        raise RuntimeError("LangChain packages are required for this pipeline. Install the langchain extras.") from exc

    docs = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "demo-1"}),
        Document(page_content="Citations improve verifiability.", metadata={"source": "demo-2"}),
    ]

    class DemoRetriever:
        def invoke(self, query: str):
            return docs[: config.get("runtime", {}).get("retriever_k", 2)]

    def _combine(inputs: Dict[str, Any]):
        question = inputs.get("input", "")
        contexts = inputs.get("context", [])
        return {
            "answer": f"Demo LangChain answer for: {question} using {len(contexts)} contexts.",
            "citations": [d.metadata.get("source", "unknown") for d in contexts],
            "contexts": [d.page_content for d in contexts],
        }

    return create_retrieval_chain(DemoRetriever(), RunnableLambda(_combine))

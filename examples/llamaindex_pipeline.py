from __future__ import annotations

from typing import Any, Dict


def create_query_engine(config: Dict[str, Any]):
    try:
        from llama_index.core import Document, VectorStoreIndex
    except Exception as exc:
        raise RuntimeError("LlamaIndex packages are required for this pipeline. Install the llamaindex extras.") from exc

    docs = [Document(text="RAG combines retrieval with generation."), Document(text="Citations improve verifiability.")]
    index = VectorStoreIndex.from_documents(docs)
    top_k = int(config.get("runtime", {}).get("retriever_top_k", 2))
    return index.as_query_engine(similarity_top_k=top_k)

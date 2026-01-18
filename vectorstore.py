# vectorstore.py
import os
from uuid import uuid4
from typing import List
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings


# ❌ DO NOT validate env vars at import time
# ❌ DO NOT create Pinecone or embeddings at import time

_embedding_model = None
_index = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def get_pinecone_index():
    global _index

    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")

        if not api_key or not index_name:
            raise RuntimeError("Pinecone env vars not set")

        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)

    return _index


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedding_model().embed_documents(texts)


def embed_query_text(query: str) -> List[float]:
    return get_embedding_model().embed_query(query)


def embed_to_pinecone(docs: List[str], metadatas: List[dict], namespace: str):
    index = get_pinecone_index()
    ids = [str(uuid4()) for _ in docs]

    for i in range(0, len(docs), 20):
        vectors = list(
            zip(
                ids[i:i+20],
                embed_texts(docs[i:i+20]),
                metadatas[i:i+20]
            )
        )
        index.upsert(vectors=vectors, namespace=namespace)


def get_relevant_context(query: str, namespace: str, top_k: int = 7) -> str:
    index = get_pinecone_index()

    res = index.query(
        vector=embed_query_text(query),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    chunks = [
        m["metadata"]["text"]
        for m in res.get("matches", [])
        if "text" in m.get("metadata", {})
    ]

    return "\n".join(chunks)

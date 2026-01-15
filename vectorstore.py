# vectorstore.py
from dotenv import load_dotenv
import os

load_dotenv()

from uuid import uuid4
from typing import List
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# ENV VARS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")
if not PINECONE_INDEX:
    raise RuntimeError("PINECONE_INDEX not set")

# Singleton embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedding_model().embed_documents(texts)

def embed_query_text(query: str) -> List[float]:
    return get_embedding_model().embed_query(query)

def embed_to_pinecone(
    docs: List[str],
    metadatas: List[dict],
    namespace: str
):
    ids = [str(uuid4()) for _ in docs]

    for i in range(0, len(docs), 20):
        batch_docs = docs[i:i+20]
        batch_meta = metadatas[i:i+20]
        batch_ids = ids[i:i+20]

        vectors = list(
            zip(batch_ids, embed_texts(batch_docs), batch_meta)
        )
        index.upsert(vectors=vectors, namespace=namespace)

def get_relevant_context(
    query: str,
    namespace: str,
    top_k: int = 7
) -> str:
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

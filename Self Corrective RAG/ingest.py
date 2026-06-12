"""
rag/ingest.py — Load listings and build FAISS vector store
"""

import json
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import SAMPLE_DATA_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL


def load_listings(path: str = SAMPLE_DATA_PATH) -> List[Document]:
    """Convert JSON listings into LangChain Documents."""
    with open(path, "r") as f:
        listings = json.load(f)

    docs: List[Document] = []
    for item in listings:
        content = (
            f"Title: {item['title']}\n"
            f"Location: {item['location']}\n"
            f"Type: {item['type']}\n"
            f"Zoning: {item['zoning']}\n"
            f"Price: ₹{item['price']:,}\n"
            f"Area: {item['area_sqft']} sqft\n"
            f"Bedrooms: {item['bedrooms']}\n"
            f"Rental Yield: {item['rental_yield_pct']}%\n"
            f"ROI: {item['roi_pct']}%\n"
            f"Description: {item['description']}"
        )
        docs.append(Document(page_content=content, metadata=item))
    return docs


def build_vector_store(docs: List[Document]) -> FAISS:
    """Embed documents and build FAISS index."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    store.save_local(VECTOR_STORE_PATH)
    print(f"[ingest] Vector store saved to {VECTOR_STORE_PATH} ({len(docs)} docs)")
    return store


def load_vector_store() -> FAISS:
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[ingest] Vector store loaded from {VECTOR_STORE_PATH}")
    return store


def get_or_build_store() -> FAISS:
    """Return existing store or build a new one."""
    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(index_file):
        return load_vector_store()
    docs = load_listings()
    return build_vector_store(docs)

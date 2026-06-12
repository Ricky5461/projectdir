"""
rag/retriever.py — Natural-language query pipeline over listings
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import TOP_K_RESULTS


class ListingRetriever:
    """Wraps FAISS store for semantic search over property listings."""

    def __init__(self, store: FAISS):
        self.store = store

    def query(self, question: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """Return top-k listings most relevant to the question."""
        return self.store.similarity_search(question, k=k)

    def query_with_score(self, question: str, k: int = TOP_K_RESULTS):
        """Return listings with similarity scores."""
        return self.store.similarity_search_with_score(question, k=k)

    def format_results(self, docs: List[Document]) -> str:
        """Format retrieved docs as readable text for LLM context."""
        if not docs:
            return "No matching listings found."
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"--- Listing {i} ---\n{doc.page_content}")
        return "\n\n".join(parts)

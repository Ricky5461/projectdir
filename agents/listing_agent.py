"""
agents/listing_agent.py — Discovers and filters property listings via RAG
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import PropBotState
from rag.ingest import get_or_build_store
from rag.retriever import ListingRetriever
from config import LLM_MODEL

_store = None


def _get_retriever() -> ListingRetriever:
    global _store
    if _store is None:
        _store = get_or_build_store()
    return ListingRetriever(_store)


def listing_agent(state: PropBotState) -> PropBotState:
    """
    ListingAgent: Takes the user query, retrieves relevant listings
    from the vector store, and summarises them using an LLM.
    """
    try:
        retriever = _get_retriever()
        docs = retriever.query(state.query)
        context = retriever.format_results(docs)

        # Extract raw listing metadata for downstream agents
        listings_found = [doc.metadata for doc in docs]

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        messages = [
            SystemMessage(
                content=(
                    "You are a real estate listing specialist. "
                    "Given the user query and retrieved listings, "
                    "summarise the most relevant properties clearly and concisely."
                )
            ),
            HumanMessage(
                content=f"User Query: {state.query}\n\nRetrieved Listings:\n{context}"
            ),
        ]
        response = llm.invoke(messages)

        return state.model_copy(
            update={
                "listings_context": response.content,
                "listings_found": listings_found,
            }
        )

    except Exception as e:
        return state.model_copy(update={"error": f"ListingAgent error: {e}"})

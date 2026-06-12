"""
agents/valuation_agent.py — Pricing analysis, ROI computation, and market benchmarking
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import PropBotState
from config import LLM_MODEL, TAVILY_API_KEY

import os


def _try_tavily_fallback(query: str) -> Optional[str]:
    """Use Tavily API for live market data if needed."""
    if not TAVILY_API_KEY:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        result = client.search(
            query=f"real estate market prices ROI {query} 2025",
            max_results=3,
        )
        snippets = [r.get("content", "") for r in result.get("results", [])]
        return "\n".join(snippets) if snippets else None
    except Exception:
        return None


def valuation_agent(state: PropBotState) -> PropBotState:
    """
    ValuationAgent: Analyses pricing trends and ROI from retrieved listings.
    Falls back to Tavily web search if local data is insufficient.
    """
    try:
        listings = state.listings_found

        # Compute basic stats from retrieved listings
        prices = [l.get("price", 0) for l in listings if l.get("price")]
        rois = [l.get("roi_pct", 0) for l in listings if l.get("roi_pct")]

        avg_price = round(sum(prices) / len(prices), 2) if prices else None
        avg_roi = round(sum(rois) / len(rois), 2) if rois else None

        # Build context for LLM
        stats_text = ""
        if avg_price:
            stats_text += f"Average Price: ₹{avg_price:,.0f}\n"
        if avg_roi:
            stats_text += f"Average ROI: {avg_roi}%\n"

        # Try Tavily fallback for live market context
        web_context = _try_tavily_fallback(state.query)
        used_fallback = web_context is not None

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        messages = [
            SystemMessage(
                content=(
                    "You are a real estate valuation expert. "
                    "Analyse the provided listings data and market context. "
                    "Give a clear pricing analysis with ROI insights. "
                    "Be specific with numbers. Never hallucinate data."
                )
            ),
            HumanMessage(
                content=(
                    f"User Query: {state.query}\n\n"
                    f"Listings Summary:\n{state.listings_context}\n\n"
                    f"Computed Stats:\n{stats_text}\n"
                    + (f"Live Market Context (Web):\n{web_context}" if web_context else "")
                )
            ),
        ]
        response = llm.invoke(messages)

        return state.model_copy(
            update={
                "valuation_report": response.content,
                "avg_price": avg_price,
                "avg_roi": avg_roi,
                "used_web_fallback": used_fallback,
            }
        )

    except Exception as e:
        return state.model_copy(update={"error": f"ValuationAgent error: {e}"})

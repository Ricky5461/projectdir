"""
agents/state.py — Shared LangGraph state passed between all agents
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PropBotState(BaseModel):
    """Shared state flowing through the LangGraph pipeline."""

    # User input
    query: str = ""

    # ListingAgent outputs
    listings_context: str = ""
    listings_found: List[dict] = Field(default_factory=list)

    # ValuationAgent outputs
    valuation_report: str = ""
    avg_price: Optional[float] = None
    avg_roi: Optional[float] = None

    # LeadAgent outputs
    final_report: str = ""

    # Fallback flag (Tavily used?)
    used_web_fallback: bool = False

    # Error tracking
    error: Optional[str] = None

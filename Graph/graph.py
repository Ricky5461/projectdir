"""
graph/workflow.py — LangGraph state graph connecting all three agents
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, END

from agents.state import PropBotState
from agents.listing_agent import listing_agent
from agents.valuation_agent import valuation_agent
from agents.lead_agent import lead_agent


def should_stop(state: PropBotState) -> str:
    """Route to END if any agent raised an error."""
    if state.error:
        return "error"
    return "continue"


def build_graph() -> Any:
    """Build and compile the PropBot LangGraph pipeline."""

    # LangGraph requires dict-based state; wrap our Pydantic model
    def listing_node(state: Dict) -> Dict:
        s = PropBotState(**state)
        result = listing_agent(s)
        return result.model_dump()

    def valuation_node(state: Dict) -> Dict:
        s = PropBotState(**state)
        result = valuation_agent(s)
        return result.model_dump()

    def lead_node(state: Dict) -> Dict:
        s = PropBotState(**state)
        result = lead_agent(s)
        return result.model_dump()

    def error_check(state: Dict) -> str:
        return "error_exit" if state.get("error") else "next"

    builder = StateGraph(dict)

    # Add nodes
    builder.add_node("listing_agent", listing_node)
    builder.add_node("valuation_agent", valuation_node)
    builder.add_node("lead_agent", lead_node)

    # Entry point
    builder.set_entry_point("listing_agent")

    # Conditional edges — stop on error, else continue
    builder.add_conditional_edges(
        "listing_agent",
        error_check,
        {"next": "valuation_agent", "error_exit": END},
    )
    builder.add_conditional_edges(
        "valuation_agent",
        error_check,
        {"next": "lead_agent", "error_exit": END},
    )
    builder.add_edge("lead_agent", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(query: str) -> PropBotState:
    """Run the full PropBot pipeline for a given query."""
    graph = get_graph()
    initial_state = PropBotState(query=query).model_dump()
    result = graph.invoke(initial_state)
    return PropBotState(**result)

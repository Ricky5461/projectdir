"""
agents/lead_agent.py — Generates structured investor reports and lead summaries
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import PropBotState
from config import LLM_MODEL


def lead_agent(state: PropBotState) -> PropBotState:
    """
    LeadAgent: Synthesises all upstream outputs into a final
    structured investor report ready for sharing or export.
    """
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

        fallback_note = (
            "\n[Note: Live market data was used to supplement local listings.]"
            if state.used_web_fallback
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a senior real estate investment analyst. "
                    "Create a professional investor report based on the research below. "
                    "Structure it with: Executive Summary, Top Properties, "
                    "Valuation Analysis, ROI Highlights, and Recommendation. "
                    "Use clear headings and bullet points."
                )
            ),
            HumanMessage(
                content=(
                    f"Investor Query: {state.query}\n\n"
                    f"Listings Research:\n{state.listings_context}\n\n"
                    f"Valuation Analysis:\n{state.valuation_report}"
                    f"{fallback_note}"
                )
            ),
        ]
        response = llm.invoke(messages)

        return state.model_copy(update={"final_report": response.content})

    except Exception as e:
        return state.model_copy(update={"error": f"LeadAgent error: {e}"})

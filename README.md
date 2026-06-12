# 🏠 PropBot — Agentic AI Platform for Real Estate

> An end-to-end multi-agent AI system for automated property discovery, pricing analysis, and investor reporting — powered by LangGraph and RAG.

**Built by:**  www.linkedin.com/in/ricky-kumar-ptiii6202
Ricky kumar

## 🚀 Overview

PropBot is an agentic AI platform that automates the full real estate research pipeline. Instead of manually browsing listings, running valuations, and compiling reports, PropBot uses specialized AI agents working together to do it all — from finding properties to generating investor-ready insights.

---

## 🤖 Multi-Agent Architecture

Built with **LangGraph**, PropBot uses a graph of specialized agents, each owning a distinct part of the workflow:

| Agent | Responsibility |
|---|---|
| `ListingAgent` | Discovers and filters property listings based on user criteria |
| `ValuationAgent` | Analyzes pricing trends, computes ROI, and benchmarks comparables |
| `LeadAgent` | Generates structured investor reports and lead summaries |

---

## 🔍 RAG Pipeline

- Indexes **50,000+ real estate listings** into a vector store
- Supports **natural-language queries** over inventory, zoning data, and ROI metrics
- Integrated **Tavily API** as a live web search fallback for real-time market data
- Achieves **zero-hallucination responses** by grounding all answers in retrieved context

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT / Claude |
| RAG & Vector Store | FAISS / Chroma + OpenAI Embeddings |
| Live Data Fallback | Tavily Search API |
| Backend | Python |

---

## ⚙️ Getting Started

```bash
git clone https://github.com/Ricky5461/projectdir.git
cd projectdir
pip install -r requirements.txt
cp .env.example .env
# Add your API keys: OPENAI_API_KEY, TAVILY_API_KEY
python main.py
```

---
## 📁 Project Structure

```
projectdir/
├── agents/
│   ├── listing_agent.py      # Property discovery & filtering
│   ├── valuation_agent.py    # Pricing analysis & ROI computation
│   └── lead_agent.py         # Investor report generation
├── rag/
│   ├── ingest.py             # Index 50K+ listings into vector store
│   └── retriever.py          # Natural-language query pipeline
├── graph/
│   └── workflow.py           # LangGraph state graph definition
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```
## 📌 Status

🟢 Active Development — Jan 2026 – Present

"""
main.py — PropBot CLI entry point
"""

import sys
from graph.workflow import run_pipeline


DEMO_QUERIES = [
    "Show me 3BHK apartments in Bangalore under ₹80 lakhs with good ROI",
    "Find commercial properties with high rental yield",
    "What are the best investment properties in Pune?",
]


def print_report(result) -> None:
    print("\n" + "=" * 60)
    print("📊 PROPBOT INVESTOR REPORT")
    print("=" * 60)

    if result.error:
        print(f"\n❌ Error: {result.error}")
        return

    if result.avg_price:
        print(f"\n💰 Average Price : ₹{result.avg_price:,.0f}")
    if result.avg_roi:
        print(f"📈 Average ROI   : {result.avg_roi}%")
    if result.used_web_fallback:
        print("🌐 Live market data (Tavily) was used to supplement results.")

    print("\n" + "-" * 60)
    print(result.final_report)
    print("=" * 60 + "\n")


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\n🏠 Welcome to PropBot — Agentic AI for Real Estate\n")
        print("Demo queries:")
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"  {i}. {q}")
        print()
        query = input("Enter your query (or press Enter for demo #1): ").strip()
        if not query:
            query = DEMO_QUERIES[0]

    print(f"\n🔍 Running pipeline for: '{query}'\n")
    print("  [1/3] ListingAgent   — retrieving properties...")
    result = run_pipeline(query)
    print("  [2/3] ValuationAgent — analysing prices & ROI...")
    print("  [3/3] LeadAgent      — generating investor report...")

    print_report(result)


if __name__ == "__main__":
    main()

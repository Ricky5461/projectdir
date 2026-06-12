import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
 
LLM_MODEL: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-3-small"

VECTOR_STORE_PATH: str = "data/faiss_index"
TOP_K_RESULTS: int = 5

# ── Data ──
SAMPLE_DATA_PATH: str = "data/listings.json"
 

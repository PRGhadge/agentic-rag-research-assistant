"""
Central config for the pipeline. Everything pulls from here so I only
need to change settings in one place.
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
OUTPUT_DIR = BASE_DIR / "output"

# --- LLM ---
# using gpt-4o-mini to keep costs low -- still good quality for summaries
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAG settings ---
# 4000 chars per chunk (~1000 tokens), 200 char overlap so sentences
# don't get cut at boundaries
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# Agentic RAG Research Assistant

A multi-agent pipeline that reads research papers, analyzes them, and generates structured summaries -- built with LangGraph and OpenAI. Includes a RAG-powered Q&A mode for asking follow-up questions about the papers.

## How it works

```
PDF papers ──→ [Reader Node] ──→ [Analyst Node] ──→ [Writer Node] ──→ summary.md
                    │                   │                  │
                    └───────── shared state dict ──────────┘
```

Three nodes run in sequence inside a LangGraph StateGraph:

1. **Reader** -- extracts title, authors, methodology, findings, and conclusions from each paper
2. **Analyst** -- evaluates methodology rigor, identifies contributions, flags limitations and gaps
3. **Writer** -- combines everything into a clean markdown summary

Each node is a plain function that reads from a shared state dictionary and writes its output back. LangGraph handles the execution order and state merging.

## Q&A Mode

After summarization, you can ask follow-up questions about the paper(s). Q&A uses RAG (Retrieval-Augmented Generation) to pull relevant passages from the papers:

1. Papers are chunked into overlapping segments (4000 chars, 200 overlap)
2. Chunks are embedded with `text-embedding-3-small` and stored in ChromaDB
3. Each question retrieves the top-5 most relevant chunks via cosine similarity
4. Chunks + summary + question are sent to the LLM for a grounded answer

Conversation history is maintained, so follow-up questions work naturally.

## Project Structure

```
├── run.py              # main entry point -- graph setup, execution, Q&A loop
├── nodes.py            # reader, analyst, writer node functions
├── state.py            # PipelineState TypedDict (shared state schema)
├── rag.py              # chunking, embedding, ChromaDB retrieval
├── config.py           # central config (model, paths, chunk settings)
├── evaluation/
│   └── evaluate.py     # summary quality evaluation (coverage, coherence, grounding)
├── data/papers/        # drop your PDFs here
└── output/
    └── summary.md      # generated summary
```

## Setup

```bash
# clone and install
git clone https://github.com/<your-username>/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# drop a PDF into data/papers/
cp your-paper.pdf data/papers/

# run the full pipeline
python run.py

# ask questions about the paper without re-running the pipeline
python run.py --qa

# evaluate summary quality
python -m evaluation.evaluate
```

## Evaluation

The evaluation module scores summaries on three metrics:

- **Coverage** -- checks if all required sections are present (deterministic, no LLM)
- **Coherence** -- LLM-as-judge rates clarity and structure on a 1-5 scale
- **Grounding** -- compares summary claims against the source paper to catch hallucinations

Results are saved to `output/evaluation_results.json`.

## Tech Stack

- **LangGraph** -- agent graph orchestration (StateGraph with typed state)
- **OpenAI GPT-4o-mini** -- LLM for all nodes and Q&A
- **ChromaDB** -- in-memory vector store for RAG retrieval
- **text-embedding-3-small** -- embedding model for chunking and search
- **pypdfium2** -- PDF text extraction

## Cost

Using `gpt-4o-mini` keeps things cheap:
- Full pipeline run: ~$0.03-0.05
- Each Q&A question: ~$0.001-0.002
- Evaluation: ~$0.01

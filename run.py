"""
Main entry point for the LangGraph research paper pipeline.

Reads PDFs, runs them through a 3-node graph (reader -> analyst -> writer),
saves the summary, and optionally enters a RAG-powered Q&A mode.

Usage:
  python run.py              # full pipeline + optional Q&A
  python run.py --qa         # Q&A only on existing summary
"""

import sys
from pathlib import Path

import pypdfium2 as pdfium
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from config import PAPERS_DIR, OUTPUT_DIR, LLM_MODEL
from nodes import reader_node, analyst_node, writer_node
from state import PipelineState
from rag import build_vector_store, retrieve


def load_paper_texts():
    """Reads all PDFs from data/papers/ and returns them as a dict."""
    paper_texts = {}
    for pdf_file in sorted(PAPERS_DIR.glob("*.pdf")):
        doc = pdfium.PdfDocument(str(pdf_file))
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        doc.close()
        paper_texts[pdf_file.name] = "\n\n".join(pages)[:80_000]
    return paper_texts


def build_graph():
    """Sets up the LangGraph pipeline: reader -> analyst -> writer.

    Each node is a function that takes state and returns updates.
    Could add conditional edges here later (e.g., skip analyst for
    short papers, or loop back to writer if quality check fails).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("reader", reader_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "reader")
    graph.add_edge("reader", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def qa_loop(summary_text, paper_texts):
    """Interactive Q&A with RAG retrieval.

    Builds a vector store from the papers, then for each question
    retrieves relevant chunks and sends them to the LLM along with
    the summary. Keeps conversation history so follow-ups work.
    """
    client = OpenAI()

    print("\nBuilding vector store for Q&A...")
    collection = build_vector_store(paper_texts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research paper assistant. Answer questions about "
                "the research paper(s) based on the summary and retrieved "
                "context. Be concise, accurate, and grounded in the source "
                "material. If the answer is not in the provided context, "
                "say so.\n\n"
                f"--- SUMMARY ---\n{summary_text}"
            ),
        }
    ]

    print("\n" + "=" * 60)
    print("  Q&A MODE - Ask questions about the paper(s)")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Q&A mode.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Exiting Q&A mode. Goodbye!")
            break

        # pull relevant chunks from the vector store
        relevant_chunks = retrieve(collection, question)
        context = "\n\n".join(relevant_chunks) if relevant_chunks else ""

        user_content = question
        if context:
            user_content = (
                f"Relevant paper excerpts:\n{context}\n\n"
                f"Question: {question}"
            )

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")


def main():
    output_file = OUTPUT_DIR / "summary.md"

    # skip the pipeline if we just want to ask questions
    if "--qa" in sys.argv:
        if not output_file.exists():
            print("No summary found at output/summary.md")
            print("Run without --qa first to generate a summary.")
            raise SystemExit(1)
        summary_text = output_file.read_text()
        paper_texts = load_paper_texts()
        qa_loop(summary_text, paper_texts)
        return

    # load papers
    paper_texts = load_paper_texts()
    if not paper_texts:
        print(f"No PDF files found in {PAPERS_DIR}")
        print("Add research papers to data/papers/ and run again.")
        raise SystemExit(1)

    print(f"Found {len(paper_texts)} paper(s):")
    for name in paper_texts:
        print(f"  - {name}")

    # build and run the graph
    print("\nRunning pipeline: reader -> analyst -> writer\n")
    app = build_graph()

    result = app.invoke({
        "paper_texts": paper_texts,
        "extraction": "",
        "analysis": "",
        "summary": "",
    })

    # save output
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file.write_text(result["summary"])
    print(f"\nSummary saved to {output_file}")
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(result["summary"])

    # offer Q&A after summarization
    print("\nWould you like to ask questions about the paper(s)?")
    try:
        enter_qa = input("Enter Q&A mode? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        enter_qa = "n"

    if enter_qa in ("y", "yes"):
        qa_loop(result["summary"], paper_texts)


if __name__ == "__main__":
    main()

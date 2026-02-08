"""
Node functions for the LangGraph pipeline.

Each node takes the shared state, calls the LLM with a specific prompt,
and returns the keys it wants to update. Kept these as plain functions
instead of classes -- easier to test and reason about.

Using the OpenAI SDK directly here rather than LangChain's ChatOpenAI
wrapper. LangGraph doesn't force you to use LangChain abstractions,
and raw SDK calls are simpler to debug.
"""

from openai import OpenAI

from config import LLM_MODEL

client = OpenAI()


def reader_node(state):
    """Reads paper text from state and extracts structured info.

    Papers are already loaded in run.py and passed via state, so no
    need for file I/O tools here -- just send the text to the LLM.
    """
    print("\n--- Reader Node: Extracting paper structure ---")

    paper_texts = state["paper_texts"]
    combined = "\n\n".join(
        f"=== {name} ===\n{text}" for name, text in paper_texts.items()
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic paper reader. You parse dense "
                    "research papers methodically and extract structured information "
                    "with precision."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Read the following research paper(s) thoroughly. "
                    "For each paper, extract:\n"
                    "1. Title\n"
                    "2. Authors\n"
                    "3. Abstract / core research question\n"
                    "4. Methodology used\n"
                    "5. Key results and findings\n"
                    "6. Conclusions\n\n"
                    "Present the extracted information clearly for each paper.\n\n"
                    f"{combined}"
                ),
            },
        ],
        temperature=0.3,
    )

    extraction = response.choices[0].message.content
    print(f"  Extraction complete ({len(extraction)} chars)")
    return {"extraction": extraction}


def analyst_node(state):
    """Takes the extraction from state and does a critical analysis.

    Reads state['extraction'] which was written by reader_node in
    the previous step. No tool calls, just LLM reasoning.
    """
    print("\n--- Analyst Node: Performing critical analysis ---")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior research analyst with deep experience "
                    "reviewing academic work across AI, ML, and computer science. "
                    "You excel at critically evaluating methodology, spotting "
                    "novel contributions, and identifying gaps or limitations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using the following extracted paper information, perform "
                    "a critical analysis:\n"
                    "1. Evaluate the methodology rigor of each paper\n"
                    "2. Identify the key novel contributions\n"
                    "3. Assess strengths and limitations\n"
                    "4. Note any gaps in the research\n"
                    "5. If multiple papers are provided, identify common themes, "
                    "contradictions, or complementary findings across them.\n\n"
                    f"{state['extraction']}"
                ),
            },
        ],
        temperature=0.3,
    )

    analysis = response.choices[0].message.content
    print(f"  Analysis complete ({len(analysis)} chars)")
    return {"analysis": analysis}


def writer_node(state):
    """Combines extraction + analysis and writes the final summary.

    This node reads both previous outputs from state and produces
    a clean markdown report with all the required sections.
    """
    print("\n--- Writer Node: Writing summary report ---")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a skilled technical writer specializing in "
                    "distilling complex research into clear, structured "
                    "summaries for a technical audience."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using the paper extraction and analysis below, write a "
                    "final structured summary report with these sections:\n\n"
                    "## Paper Overview\n"
                    "Title, authors, and publication context.\n\n"
                    "## Problem Statement\n"
                    "What problem each paper addresses and why it matters.\n\n"
                    "## Methodology\n"
                    "How the research was conducted.\n\n"
                    "## Key Findings\n"
                    "The most important results and discoveries.\n\n"
                    "## Limitations & Future Work\n"
                    "Acknowledged limitations and open questions.\n\n"
                    "## Cross-Paper Themes\n"
                    "(If multiple papers) Common themes and disagreements.\n\n"
                    "## Key Takeaways\n"
                    "3-5 bullet points capturing the most important insights.\n\n"
                    "Write in clear, accessible language suitable for a "
                    "technical audience.\n\n"
                    f"--- EXTRACTION ---\n{state['extraction']}\n\n"
                    f"--- ANALYSIS ---\n{state['analysis']}"
                ),
            },
        ],
        temperature=0.3,
    )

    summary = response.choices[0].message.content
    print(f"  Summary complete ({len(summary)} chars)")
    return {"summary": summary}

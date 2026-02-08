"""
Evaluation framework for summary quality.

Runs three checks against the generated summary:
  1. Coverage  -- are all required sections present? (no LLM, just string matching)
  2. Coherence -- is the writing clear and well-structured? (LLM-as-judge, 1-5 scale)
  3. Grounding -- are the claims actually in the source paper? (hallucination detector)

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --summary output/summary.md --papers data/papers/
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

from config import PAPERS_DIR, OUTPUT_DIR

# cheaper model for judging -- evaluation is simpler than summarization
JUDGE_MODEL = "gpt-4o-mini"

# sections the writer node is supposed to include
REQUIRED_SECTIONS = [
    "Paper Overview",
    "Problem Statement",
    "Methodology",
    "Key Findings",
    "Limitations & Future Work",
    "Key Takeaways",
]


def evaluate_coverage(summary: str) -> dict:
    """Checks if all required section headings are in the summary.
    No LLM needed -- just string matching.
    """
    found = []
    missing = []
    for section in REQUIRED_SECTIONS:
        if section.lower() in summary.lower():
            found.append(section)
        else:
            missing.append(section)

    score = len(found) / len(REQUIRED_SECTIONS)
    return {
        "metric": "coverage",
        "score": round(score, 2),
        "found_sections": found,
        "missing_sections": missing,
    }


def evaluate_coherence(client: OpenAI, summary: str) -> dict:
    """Has GPT-4o-mini rate the summary's structure and clarity 1-5.
    Uses temperature=0 so scores are reproducible across runs.
    """
    prompt = (
        "You are evaluating a research paper summary for coherence.\n\n"
        "Rate the following summary on a scale of 1-5 for:\n"
        "- Logical flow between sections\n"
        "- Clarity of language\n"
        "- Consistent level of detail\n"
        "- Proper use of headings and structure\n\n"
        "Return a JSON object with:\n"
        '  "score": <float 1-5>,\n'
        '  "reasoning": "<brief explanation>"\n\n'
        f"Summary to evaluate:\n\n{summary}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    return {
        "metric": "coherence",
        "score": round(result["score"] / 5, 2),
        "raw_score": result["score"],
        "reasoning": result["reasoning"],
    }


def evaluate_grounding(client: OpenAI, summary: str, source_text: str) -> dict:
    """Hallucination detector -- compares summary claims against the source.
    Gives the judge both texts and asks it to flag anything not supported.
    """
    # cap source text so we don't blow the context window
    max_source = 30_000
    if len(source_text) > max_source:
        source_text = source_text[:max_source] + "\n[...truncated...]"

    prompt = (
        "You are evaluating whether a research paper summary is factually grounded "
        "in the source text.\n\n"
        "Check if the key claims in the summary can be traced back to the source.\n"
        "Flag any claims that appear fabricated or unsupported.\n\n"
        "Return a JSON object with:\n"
        '  "score": <float 1-5>,\n'
        '  "supported_claims": <int>,\n'
        '  "unsupported_claims": <int>,\n'
        '  "flagged": ["<list of unsupported claims if any>"]\n\n'
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"SUMMARY TO EVALUATE:\n{summary}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    return {
        "metric": "grounding",
        "score": round(result["score"] / 5, 2),
        "supported_claims": result.get("supported_claims", 0),
        "unsupported_claims": result.get("unsupported_claims", 0),
        "flagged": result.get("flagged", []),
    }


def load_source_text(papers_dir: Path) -> str:
    """Extracts text from all PDFs for the grounding check."""
    import pypdfium2 as pdfium

    all_text = []
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        pdf = pdfium.PdfDocument(str(pdf_path))
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        pdf.close()
        all_text.append(f"=== {pdf_path.name} ===\n" + "\n".join(pages))

    return "\n\n".join(all_text)


def run_evaluation(summary_path: Path, papers_dir: Path):
    """Runs all three metrics and saves results as JSON."""
    summary = summary_path.read_text()
    if not summary.strip():
        print("Error: Summary file is empty.")
        raise SystemExit(1)

    client = OpenAI()

    print("Running evaluation...\n")

    # 1. coverage (free, no API call)
    coverage = evaluate_coverage(summary)
    print(f"Coverage:  {coverage['score']:.0%}")
    if coverage["missing_sections"]:
        print(f"  Missing: {', '.join(coverage['missing_sections'])}")

    # 2. coherence (1 API call)
    coherence = evaluate_coherence(client, summary)
    print(f"Coherence: {coherence['score']:.0%} ({coherence['raw_score']}/5)")
    print(f"  {coherence['reasoning']}")

    # 3. grounding (1 API call)
    source_text = load_source_text(papers_dir)
    if source_text.strip():
        grounding = evaluate_grounding(client, summary, source_text)
        print(f"Grounding: {grounding['score']:.0%}")
        print(f"  Supported: {grounding['supported_claims']}, "
              f"Unsupported: {grounding['unsupported_claims']}")
        if grounding["flagged"]:
            print("  Flagged claims:")
            for claim in grounding["flagged"]:
                print(f"    - {claim}")
    else:
        grounding = {"metric": "grounding", "score": None, "note": "No source text"}
        print("Grounding: skipped (no source PDFs found)")

    # overall average
    scores = [r["score"] for r in [coverage, coherence, grounding] if r["score"] is not None]
    overall = sum(scores) / len(scores) if scores else 0
    print(f"\nOverall:   {overall:.0%}")

    # save for tracking over time
    results = {
        "coverage": coverage,
        "coherence": coherence,
        "grounding": grounding,
        "overall_score": round(overall, 2),
    }
    results_path = summary_path.parent / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate research paper summary quality")
    parser.add_argument("--summary", type=Path, default=OUTPUT_DIR / "summary.md")
    parser.add_argument("--papers", type=Path, default=PAPERS_DIR)
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Summary not found at {args.summary}")
        print("Run the pipeline first: python run.py")
        raise SystemExit(1)

    run_evaluation(args.summary, args.papers)


if __name__ == "__main__":
    main()

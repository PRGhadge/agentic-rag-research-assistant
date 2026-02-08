"""
Shared state definition for the pipeline.

Every node in the graph reads from and writes to this state dict.
I went with TypedDict over a dataclass because LangGraph expects it
and it keeps things simple -- each node just returns the keys it
wants to update, and LangGraph handles merging them back in.
"""

from typing import TypedDict


class PipelineState(TypedDict):
    paper_texts: dict[str, str]  # raw PDF text keyed by filename
    extraction: str               # structured info pulled out by the reader
    analysis: str                 # critical analysis from the analyst
    summary: str                  # final markdown report

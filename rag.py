"""
RAG module for the Q&A mode.

Handles chunking paper text, embedding into ChromaDB, and retrieving
relevant passages when the user asks questions.

I only use RAG for Q&A, not for the main pipeline -- the full paper
text fits in gpt-4o-mini's 128k context window, so RAG would just
add overhead there. But for Q&A it makes sense because you don't want
to stuff the entire paper into every single question prompt.
"""

import chromadb
from openai import OpenAI

from config import CHUNK_SIZE, CHUNK_OVERLAP

client = OpenAI()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks.

    Overlap ensures we don't lose context at chunk boundaries --
    the end of one chunk overlaps with the start of the next.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_vector_store(paper_texts):
    """Chunks all papers, embeds them, and stores in ChromaDB.

    Does a single batch embedding call for all chunks rather than
    one-by-one -- way cheaper and faster. Returns the collection
    so we can query it later in the Q&A loop.
    """
    chroma_client = chromadb.Client()  # in-memory, lives for the session
    collection = chroma_client.create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filename, text in paper_texts.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{i}")
            all_metadata.append({"source": filename, "chunk_index": i})

    if not all_chunks:
        return collection

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=all_chunks,
    )
    embeddings = [item.embedding for item in response.data]

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )

    print(f"  Indexed {len(all_chunks)} chunks from {len(paper_texts)} paper(s)")
    return collection


def retrieve(collection, question, top_k=5):
    """Finds the most relevant chunks for a given question.

    Embeds the question with the same model used for indexing,
    then queries ChromaDB for the top-k nearest neighbors.
    5 chunks is usually enough context without blowing up token costs.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results["documents"][0] if results["documents"] else []

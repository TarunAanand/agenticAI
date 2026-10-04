import math
import re
from collections import Counter
from io import BytesIO
from typing import TypedDict

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CHUNKS_PER_DOCUMENT = 2_000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.12
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
WORD_PATTERN = re.compile(r"[a-z0-9]+")


class TextChunk(TypedDict):
    source: str
    content: str


class RetrievedChunk(TextChunk):
    score: float


class MaterialError(ValueError):
    """A user-facing validation or extraction error for an uploaded material."""


def _split_text(text: str, source: str) -> list[TextChunk]:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        content = normalized[start:end].strip()
        if content:
            chunks.append({"source": source, "content": content})
        if end >= len(normalized):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def extract_and_chunk(name: str, data: bytes) -> list[TextChunk]:
    """Extract supported text documents and split them into overlapping chunks."""
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    extension = f".{suffix}"
    if extension not in SUPPORTED_EXTENSIONS:
        raise MaterialError("Unsupported file type. Upload a PDF, TXT, or Markdown file.")
    if not data:
        raise MaterialError("The selected file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise MaterialError("The file is too large. The maximum upload size is 8 MB.")

    extracted: list[TextChunk] = []
    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise MaterialError("This PDF is password-protected and cannot be indexed.")
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                extracted.extend(_split_text(page_text, f"{name} · page {page_number}"))
        except MaterialError:
            raise
        except (PdfReadError, EOFError, OSError, ValueError) as error:
            raise MaterialError("This PDF could not be read. Check that it is a valid PDF.") from error
    else:
        text = data.decode("utf-8-sig", errors="replace")
        extracted = _split_text(text, name)

    if not extracted:
        raise MaterialError(
            "No selectable text was found. Scanned PDFs need OCR, which is not included yet."
        )
    if len(extracted) > MAX_CHUNKS_PER_DOCUMENT:
        raise MaterialError(
            f"This document is too long to index at once (over {MAX_CHUNKS_PER_DOCUMENT} text chunks)."
        )
    return extracted


def retrieve_chunks(
    query: str,
    chunks: list[TextChunk],
    top_k: int = TOP_K,
    minimum_score: float = MIN_RETRIEVAL_SCORE,
) -> list[RetrievedChunk]:
    """Rank locally with TF-IDF and drop weak lexical matches."""
    if not 0 <= minimum_score <= 1:
        raise ValueError("Minimum retrieval score must be between 0 and 1.")
    query_terms = Counter(WORD_PATTERN.findall(query.lower()))
    if not query_terms or not chunks or top_k <= 0:
        return []

    tokenized = [Counter(WORD_PATTERN.findall(chunk["content"].lower())) for chunk in chunks]
    document_frequency = Counter(
        term for terms in tokenized for term in terms.keys()
    )
    document_count = len(tokenized)

    def weight(term: str, count: int) -> float:
        inverse_frequency = math.log((document_count + 1) / (document_frequency[term] + 1)) + 1
        return (1 + math.log(count)) * inverse_frequency

    query_vector = {term: weight(term, count) for term, count in query_terms.items()}
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    ranked: list[RetrievedChunk] = []

    for chunk, terms in zip(chunks, tokenized):
        matching = query_vector.keys() & terms.keys()
        if not matching:
            continue
        chunk_vector = {term: weight(term, terms[term]) for term in matching}
        dot_product = sum(query_vector[term] * chunk_vector[term] for term in matching)
        chunk_norm = math.sqrt(
            sum(weight(term, count) ** 2 for term, count in terms.items())
        )
        score = dot_product / (query_norm * chunk_norm) if chunk_norm else 0.0
        if score >= minimum_score:
            ranked.append({**chunk, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved excerpts with source IDs for the grounded chat prompt."""
    return "\n\n".join(
        f"[{index}] Source: {chunk['source']}\n{chunk['content']}"
        for index, chunk in enumerate(chunks, start=1)
    )

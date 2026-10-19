import re
from typing import Dict, List, Sequence, Union

from bs4 import BeautifulSoup

from datlas.crawler import fetch_web_context


def build_country_count_notes(question: str, sources: Sequence[Dict[str, str]]) -> List[str]:
    """Preserve explicit UN-member versus broader sovereign-state counts from sources."""
    if not re.search(r"\bhow many countries\b", question, re.IGNORECASE):
        return []

    notes = []
    count_pattern = re.compile(
        r"(?:lists?|includes)\s+(\d+)\s+sovereign states?\s*\((\d+)\s+of which are member states of (?:the )?United Nations\)",
        re.IGNORECASE,
    )
    for index, source in enumerate(sources, start=1):
        match = count_pattern.search(str(source.get("text") or ""))
        if match:
            total, un_members = match.groups()
            notes.append(
                f"Source [{index}] states that {total} sovereign states are listed, "
                f"{un_members} of which are UN member countries. For a question asking "
                f"how many countries, report {un_members} UN member countries and clarify "
                f"that {total} is the broader list total."
            )
    return notes


def build_country_count_answer(question: str, sources: Sequence[Dict[str, str]]) -> str | None:
    """Correct country-count wording when a source distinguishes UN members from a broader list."""
    if not re.search(r"\bhow many countries\b", question, re.IGNORECASE):
        return None

    count_pattern = re.compile(
        r"(?:lists?|includes)\s+(\d+)\s+sovereign states?\s*\((\d+)\s+of which are member states of (?:the )?United Nations\)",
        re.IGNORECASE,
    )
    region_match = re.search(r"\bin\s+([^?.!]+)", question, re.IGNORECASE)
    region = region_match.group(1).strip() if region_match else "the region"
    for index, source in enumerate(sources, start=1):
        match = count_pattern.search(str(source.get("text") or ""))
        if match:
            total, un_members = match.groups()
            return (
                f"The source identifies {un_members} United Nations member countries in {region}. "
                f"Its broader list includes {total} sovereign states; those counts use different criteria [{index}]."
            )
    return None


def build_citation_list(citations: Sequence[Dict[str, str]]) -> List[str]:
    """Render a clean list of citations with titles and URLs."""
    formatted: List[str] = []
    for citation in citations:
        title = str(citation.get("title") or "Source")
        url = str(citation.get("url") or "")
        snippet = str(citation.get("snippet") or "")
        line = f"[{title}]({url})"
        if snippet:
            line = f"{line} — {snippet[:180]}"
        formatted.append(line)
    return formatted


def sanitize_text(raw_text: str) -> str:
    """Clean a page or snippet down to readable, compact text."""
    if not raw_text:
        return ""
    soup = BeautifulSoup(raw_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """Split a large text block into overlapping chunks."""
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def select_context_excerpt(question: str, text: str, max_chars: int) -> str:
    """Keep the most query-relevant sentences within a prompt budget."""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return ""
    stop_words = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "with"}
    query_terms = {
        term.lower()
        for term in re.findall(r"[\w-]+", question)
        if len(term) > 2 and term.lower() not in stop_words
    }
    if not query_terms:
        return text[:max_chars]

    scored = []
    for index, sentence in enumerate(sentences):
        terms = {term.lower() for term in re.findall(r"[\w-]+", sentence)}
        overlap = query_terms & terms
        if overlap:
            scored.append((len(overlap) / len(query_terms), index, sentence))
    if not scored:
        return text[:max_chars]

    chosen = []
    used = 0
    for _, index, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = sentence if len(sentence) <= remaining else sentence[:remaining].rsplit(" ", 1)[0]
        if excerpt:
            chosen.append((index, excerpt))
            used += len(excerpt) + 1
    return " ".join(sentence for _, sentence in sorted(chosen))


def rank_documents(query: str, documents: Sequence[Union[str, Dict[str, str]]], top_k: int = 3) -> List[str]:
    """Return relevant document chunks by simple token-overlap scoring."""
    if not documents:
        return []

    normalized_docs: List[str] = []
    for doc in documents:
        if isinstance(doc, dict):
            normalized_docs.append(str(doc.get("text") or ""))
        else:
            normalized_docs.append(str(doc))

    query_terms = {term.lower() for term in re.findall(r"\w+", query)}
    if not query_terms:
        return [doc for doc in normalized_docs[:top_k] if doc]

    scored = []
    for doc in normalized_docs:
        if not doc:
            continue
        doc_terms = re.findall(r"\w+", doc.lower())
        score = sum(doc_terms.count(term) for term in query_terms)
        scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def build_rag_prompt(question: str, snippets: Sequence[Union[str, Dict[str, str]]]) -> str:
    """Create a grounded prompt using retrieved web snippets."""
    cleaned = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            cleaned.append(str(snippet.get("text") or ""))
        else:
            cleaned.append(str(snippet))
    cleaned = [snippet for snippet in cleaned if snippet]

    if not cleaned:
        return f"Answer the following question using your best judgment.\n\nQuestion: {question}\nAnswer:"

    context = "\n\n".join(f"[Source] {snippet}" for snippet in cleaned)
    return (
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
        "If the answer is not present, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

from __future__ import annotations

REFORMULATION_SYSTEM_PROMPT = """\
You are a search query expert. Your task is to reformulate a user's question \
into a high-quality dense-vector search query and extract structured metadata \
from it.

Rules:
- Resolve all pronouns (he/she/it/they) using context if provided.
- Remove filler words and keep only semantically rich content.
- Do NOT add information that is not implied by the question.
- Return ONLY valid JSON matching the provided schema. No markdown fences.
"""

REFORMULATION_USER_TEMPLATE = """\
User question: {question}

Return JSON with these exact keys:
{{
  "refined_query": "<string>",
  "search_terms": ["<term1>", "<term2>"],
  "intent": "<string>",
  "requires_multiple_chunks": <true|false>
}}
"""

# ─────────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a precise knowledge assistant. Answer questions ONLY using the \
provided document excerpts. Follow these rules strictly:

1. Base your answer exclusively on the provided context. Do not use external \
   knowledge.
2. If the context does not contain enough information to answer, respond with \
   exactly: "The information requested is not available in the provided \
   documents."
3. Cite specific document names (e.g. "According to <filename>...") when \
   making factual claims.
4. Keep answers concise and factual. Avoid speculation.
5. If multiple documents agree or disagree, note that explicitly.
"""

ANSWER_USER_TEMPLATE = """\
Question: {question}

---
Document Excerpts:
{context}
---

Provide a grounded answer based solely on the excerpts above.
"""


def build_context_block(evidence_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Args:
        evidence_chunks: List of dicts with 'filename', 'chunk_index', 'text'.

    Returns:
        Multi-line string with numbered excerpts.
    """
    parts: list[str] = []
    for i, chunk in enumerate(evidence_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        chunk_idx = chunk.get("chunk_index", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[{i}] Source: {filename} (chunk {chunk_idx})\n{text}")
    return "\n\n".join(parts)

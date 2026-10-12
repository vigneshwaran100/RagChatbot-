"""
Evaluation suite: 10+ questions covering answerable, unanswerable, and
multi-chunk scenarios. Run against a live API instance.

Usage:
    # Start the API first, then:
    pytest tests/evaluation/test_questions.py -v --api-url=http://localhost:8000

These tests require sample documents to be ingested first.
See sample_documents/ directory.
"""
from __future__ import annotations

import pytest
import httpx

BASE_URL = "http://localhost:8000/api/v1"

# ─────────────────────────────────────────────────────────────────────────────
# Test question bank
# Format: (question, expect_answerable, category)
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_QUESTIONS = [
    # ── Answerable: Single chunk ──────────────────────────────────────────────
    (
        "What is Python primarily used for?",
        True,
        "answerable_single",
    ),
    (
        "What does RAG stand for in AI?",
        True,
        "answerable_single",
    ),
    (
        "What is the purpose of vector embeddings?",
        True,
        "answerable_single",
    ),
    (
        "What is semantic search?",
        True,
        "answerable_single",
    ),
    (
        "What is a large language model?",
        True,
        "answerable_single",
    ),

    # ── Answerable: Multi-chunk ───────────────────────────────────────────────
    (
        "Compare the benefits and limitations of RAG versus fine-tuning.",
        True,
        "answerable_multi",
    ),
    (
        "Describe the full pipeline from document ingestion to answer generation.",
        True,
        "answerable_multi",
    ),

    # ── Unanswerable: Not in documents ───────────────────────────────────────
    (
        "What is the GDP of Iceland in 2023?",
        False,
        "unanswerable",
    ),
    (
        "Who won the FIFA World Cup in 2026?",
        False,
        "unanswerable",
    ),
    (
        "What is the boiling point of mercury in Fahrenheit?",
        False,
        "unanswerable",
    ),
    (
        "What is the latest version of the Linux kernel?",
        False,
        "unanswerable",
    ),

    # ── Edge cases ────────────────────────────────────────────────────────────
    (
        "Give me a step-by-step guide to building a rocket ship.",
        False,
        "edge_case",
    ),
]


@pytest.mark.parametrize("question,expect_answerable,category", EVALUATION_QUESTIONS)
def test_evaluation_question(question: str, expect_answerable: bool, category: str):
    """
    Send each evaluation question to the live API and verify:
    - HTTP 200 response
    - is_answerable matches expectation
    - Answerable responses include at least 1 evidence item
    - Unanswerable responses return the standard "not available" message
    """
    try:
        response = httpx.post(
            f"{BASE_URL}/query",
            json={"question": question},
            timeout=60.0,
        )
    except httpx.ConnectError:
        pytest.skip("API not running at localhost:8000 — start it first.")

    assert response.status_code == 200, (
        f"[{category}] '{question}' → HTTP {response.status_code}: {response.text}"
    )

    body = response.json()

    assert "is_answerable" in body
    assert "answer" in body
    assert "evidence" in body

    if expect_answerable:
        assert body["is_answerable"] is True, (
            f"[{category}] Expected answerable but got: {body['answer']}"
        )
        assert len(body["evidence"]) >= 1, (
            f"[{category}] Answerable response must include evidence."
        )
    else:
        assert body["is_answerable"] is False, (
            f"[{category}] Expected unanswerable but LLM answered: {body['answer']}"
        )
        assert "not available" in body["answer"].lower(), (
            f"[{category}] Unanswerable response should say 'not available'."
        )
        assert body["evidence"] == [], (
            f"[{category}] Unanswerable response must have empty evidence."
        )

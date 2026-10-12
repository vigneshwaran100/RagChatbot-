"""
RAGAS Evaluation Pipeline for the RAG Knowledge Assistant.

This script:
1. Loads questions + ground truths from eval_dataset.json
2. Calls the live /query API for each question
3. Collects the generated answer and retrieved context excerpts
4. Runs RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
5. Prints a scored report and saves results to eval_report.md

Usage:
    # Ensure the API is running and sample documents are ingested first
    python tests/evaluation/ragas_eval.py

    # Point to a different API URL
    python tests/evaluation/ragas_eval.py --api-url http://localhost:8000

    # Save results to a custom path
    python tests/evaluation/ragas_eval.py --output tests/evaluation/eval_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── RAGAS imports ─────────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset
except ImportError:
    print(
        "\n[ERROR] RAGAS or datasets package not installed.\n"
        "Run: pip install ragas datasets\n"
    )
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
DEFAULT_REPORT_PATH = Path(__file__).parent / "eval_report.md"
DEFAULT_API_URL = "http://localhost:8000"
NOT_IN_DOCS_MARKER = "NOT_IN_DOCUMENTS"

# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_api(api_url: str, question: str, timeout: int = 60) -> dict | None:
    """Call POST /api/v1/query and return the JSON body, or None on failure."""
    try:
        resp = httpx.post(
            f"{api_url}/api/v1/query",
            json={"question": question, "top_k": 5},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(f"  ✗ Cannot connect to API at {api_url}. Is it running?")
        return None
    except Exception as exc:
        print(f"  ✗ API error for question '{question[:50]}': {exc}")
        return None


def collect_responses(api_url: str, dataset: list[dict]) -> list[dict]:
    """
    For each question in the dataset, call the API and collect:
      - question
      - answer
      - contexts (list of excerpt strings from evidence)
      - ground_truth
      - is_answerable
    """
    print(f"\n📡 Calling API at {api_url} for {len(dataset)} questions...\n")
    results = []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        is_unanswerable = ground_truth == NOT_IN_DOCS_MARKER

        print(f"  [{i:02d}/{len(dataset)}] {question[:70]}")

        api_response = call_api(api_url, question)
        if api_response is None:
            print("         → Skipped (API unavailable)")
            continue

        answer = api_response.get("answer", "")
        is_answerable = api_response.get("is_answerable", False)
        evidence = api_response.get("evidence", [])
        contexts = [e["excerpt"] for e in evidence if e.get("excerpt")]

        print(
            f"         → is_answerable={is_answerable}, "
            f"contexts={len(contexts)}, "
            f"provider={api_response.get('provider_used', '?')}"
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved."],
                "ground_truth": ground_truth if not is_unanswerable else answer,
                "is_answerable": is_answerable,
                "expected_answerable": not is_unanswerable,
            }
        )

    return results


def split_by_answerability(
    responses: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Separate answerable and unanswerable responses for separate scoring."""
    answerable = [r for r in responses if r["is_answerable"]]
    unanswerable = [r for r in responses if not r["is_answerable"]]
    return answerable, unanswerable


def run_ragas(responses: list[dict]) -> dict:
    """
    Build a HuggingFace Dataset and run RAGAS evaluation.

    Metrics:
      - faithfulness:        Is the answer grounded in the retrieved context?
      - answer_relevancy:    Is the answer relevant to the question?
      - context_precision:   Are retrieved chunks relevant to the question?
      - context_recall:      Does the context cover the ground truth answer?

    Returns dict of metric_name → score (0.0–1.0).
    """
    if not responses:
        return {}

    hf_dataset = Dataset.from_dict(
        {
            "question":     [r["question"] for r in responses],
            "answer":       [r["answer"] for r in responses],
            "contexts":     [r["contexts"] for r in responses],
            "ground_truth": [r["ground_truth"] for r in responses],
        }
    )

    print("\n🔬 Running RAGAS evaluation (this may take a minute)...\n")

    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {
        "faithfulness":       round(float(result["faithfulness"]), 4),
        "answer_relevancy":   round(float(result["answer_relevancy"]), 4),
        "context_precision":  round(float(result["context_precision"]), 4),
        "context_recall":     round(float(result["context_recall"]), 4),
        "ragas_score":        round(
            float(
                (
                    result["faithfulness"]
                    + result["answer_relevancy"]
                    + result["context_precision"]
                    + result["context_recall"]
                )
                / 4
            ),
            4,
        ),
    }


def score_unanswerable(unanswerable: list[dict]) -> dict:
    """
    For unanswerable questions, check:
    - Did the system correctly return is_answerable=False?
    - Does the answer contain the "not available" phrase?
    """
    total = len(unanswerable)
    if total == 0:
        return {"total": 0, "correct_refusals": 0, "refusal_rate": 1.0}

    correct = sum(
        1
        for r in unanswerable
        if not r["is_answerable"] and "not available" in r["answer"].lower()
    )
    return {
        "total": total,
        "correct_refusals": correct,
        "refusal_rate": round(correct / total, 4),
    }


def print_report(
    ragas_scores: dict,
    unanswerable_scores: dict,
    total_questions: int,
    answerable_count: int,
) -> None:
    """Print a formatted terminal report."""
    print("\n" + "=" * 60)
    print("           RAGAS EVALUATION REPORT")
    print("=" * 60)

    print(f"\n📋 Dataset Summary")
    print(f"   Total questions      : {total_questions}")
    print(f"   Answerable questions : {answerable_count}")
    print(f"   Unanswerable         : {unanswerable_scores['total']}")

    if ragas_scores:
        print(f"\n📊 RAGAS Metrics (Answerable Questions)")
        print(f"   Faithfulness         : {ragas_scores['faithfulness']:.4f}  {'✅' if ragas_scores['faithfulness'] >= 0.7 else '⚠️'}")
        print(f"   Answer Relevancy     : {ragas_scores['answer_relevancy']:.4f}  {'✅' if ragas_scores['answer_relevancy'] >= 0.7 else '⚠️'}")
        print(f"   Context Precision    : {ragas_scores['context_precision']:.4f}  {'✅' if ragas_scores['context_precision'] >= 0.7 else '⚠️'}")
        print(f"   Context Recall       : {ragas_scores['context_recall']:.4f}  {'✅' if ragas_scores['context_recall'] >= 0.7 else '⚠️'}")
        print(f"   ─────────────────────────────────")
        print(f"   Overall RAGAS Score  : {ragas_scores['ragas_score']:.4f}  {'✅ PASS' if ragas_scores['ragas_score'] >= 0.7 else '❌ FAIL'}")

    print(f"\n🚫 Unanswerable Question Handling")
    print(f"   Correct refusals     : {unanswerable_scores['correct_refusals']}/{unanswerable_scores['total']}")
    print(f"   Refusal rate         : {unanswerable_scores['refusal_rate']:.4f}  {'✅' if unanswerable_scores['refusal_rate'] >= 0.8 else '⚠️'}")

    print("\n" + "=" * 60)


def save_report(
    report_path: Path,
    ragas_scores: dict,
    unanswerable_scores: dict,
    responses: list[dict],
    total_questions: int,
    answerable_count: int,
) -> None:
    """Save a full Markdown evaluation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# RAG Knowledge Assistant — Evaluation Report",
        "",
        f"> Generated: {now}",
        "",
        "---",
        "",
        "## Dataset Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total questions | {total_questions} |",
        f"| Answerable questions | {answerable_count} |",
        f"| Unanswerable questions | {unanswerable_scores['total']} |",
        "",
    ]

    if ragas_scores:
        def badge(score: float) -> str:
            return "✅" if score >= 0.7 else "⚠️"

        lines += [
            "## RAGAS Metrics (Answerable Questions)",
            "",
            "| Metric | Score | Status | Description |",
            "|---|---|---|---|",
            f"| Faithfulness | {ragas_scores['faithfulness']:.4f} | {badge(ragas_scores['faithfulness'])} | Answer grounded in retrieved context |",
            f"| Answer Relevancy | {ragas_scores['answer_relevancy']:.4f} | {badge(ragas_scores['answer_relevancy'])} | Answer relevant to the question |",
            f"| Context Precision | {ragas_scores['context_precision']:.4f} | {badge(ragas_scores['context_precision'])} | Retrieved chunks are relevant |",
            f"| Context Recall | {ragas_scores['context_recall']:.4f} | {badge(ragas_scores['context_recall'])} | Context covers ground truth |",
            f"| **Overall RAGAS Score** | **{ragas_scores['ragas_score']:.4f}** | {badge(ragas_scores['ragas_score'])} | Mean of all 4 metrics |",
            "",
        ]
    else:
        lines += ["## RAGAS Metrics", "", "_No answerable questions evaluated._", ""]

    lines += [
        "## Unanswerable Question Handling",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total unanswerable questions | {unanswerable_scores['total']} |",
        f"| Correctly refused (is_answerable=false) | {unanswerable_scores['correct_refusals']} |",
        f"| Refusal rate | {unanswerable_scores['refusal_rate']:.4f} |",
        "",
        "---",
        "",
        "## Per-Question Results",
        "",
        "| # | Question | Answerable | RAGAS Faithfulness | Contexts |",
        "|---|---|---|---|---|",
    ]

    for i, r in enumerate(responses, start=1):
        answerable_str = "✅ Yes" if r["is_answerable"] else "🚫 No"
        ctx_count = len([c for c in r["contexts"] if c != "No context retrieved."])
        lines.append(
            f"| {i} | {r['question'][:60]} | {answerable_str} | — | {ctx_count} chunk(s) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | How RAGAS Calculates It |",
        "|---|---|",
        "| **Faithfulness** | Claims in the answer are verified against retrieved context using an LLM judge. Score = verified_claims / total_claims |",
        "| **Answer Relevancy** | Reverse-engineers questions from the answer and measures similarity to the original question |",
        "| **Context Precision** | Measures what fraction of retrieved chunks were actually relevant to answering (ranks signal-to-noise) |",
        "| **Context Recall** | Checks how many statements in the ground truth answer are attributable to the retrieved context |",
        "",
        "## Limitations",
        "",
        "- RAGAS uses an LLM judge internally (defaults to OpenAI). Set `OPENAI_API_KEY` if scores show errors.",
        "- Context recall requires a ground truth answer — unanswerable questions are excluded from RAGAS scoring.",
        "- Scores are sensitive to the quality of `ground_truth` strings in `eval_dataset.json`.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG API.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running API")
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH), help="Output path for Markdown report")
    parser.add_argument("--dataset", default=str(EVAL_DATASET_PATH), help="Path to eval_dataset.json")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(Path(args.dataset))
    print(f"✅ Loaded {len(dataset)} questions from {args.dataset}")

    # Call the live API
    responses = collect_responses(args.api_url, dataset)
    if not responses:
        print("\n[ERROR] No API responses collected. Exiting.")
        sys.exit(1)

    # Split by answerability
    answerable, unanswerable = split_by_answerability(responses)
    print(f"\n✅ Collected {len(answerable)} answerable, {len(unanswerable)} unanswerable responses")

    # Score unanswerable refusals
    unanswerable_scores = score_unanswerable(unanswerable)

    # Run RAGAS on answerable only
    ragas_scores = {}
    if answerable:
        ragas_scores = run_ragas(answerable)
    else:
        print("\n⚠️  No answerable responses — skipping RAGAS scoring.")

    # Print terminal report
    print_report(ragas_scores, unanswerable_scores, len(responses), len(answerable))

    # Save Markdown report
    save_report(
        Path(args.output),
        ragas_scores,
        unanswerable_scores,
        responses,
        len(responses),
        len(answerable),
    )


if __name__ == "__main__":
    main()

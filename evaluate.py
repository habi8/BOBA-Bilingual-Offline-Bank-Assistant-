"""Evaluate the local assistant against reviewed question labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from assistant import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    BankAssistant,
    detect_reply_language,
    is_idk,
    ollama_content,
)

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "test-questions.csv"
LABELS_PATH = ROOT / "labels.csv"


def load_inputs(questions_path: Path, labels_path: Path) -> pd.DataFrame:
    questions = pd.read_csv(questions_path, dtype={"id": str}).fillna("")
    labels = pd.read_csv(labels_path, dtype={"id": str, "expected_faq_id": str}).fillna("")
    if not {"id", "question"}.issubset(questions.columns):
        raise ValueError("Test questions CSV must have id and question columns.")
    required = {"id", "question", "language", "answerable", "expected_faq_id"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels CSV must have columns: {', '.join(sorted(required))}")
    if len(questions) != 40:
        raise ValueError(f"Expected exactly 40 test questions; found {len(questions)}.")
    merged = questions.merge(labels, on=["id", "question"], how="left", validate="one_to_one")
    if merged["language"].isna().any() or (merged["language"] == "").any():
        raise ValueError("Labels do not match every test question; review labels.csv.")
    invalid = set(merged["language"]) - {"English", "Bangla", "Banglish"}
    if invalid or not set(merged["answerable"]).issubset({"yes", "no"}):
        raise ValueError("Labels contain an invalid language or answerable value.")
    for row in merged.itertuples(index=False):
        if (row.answerable == "yes") != bool(row.expected_faq_id):
            raise ValueError(f"Invalid expected FAQ ID for test question {row.id}.")
    return merged


def judge_correctness(
    assistant: BankAssistant,
    question: str,
    expected_id: str,
    answer: str,
    judge_model: str = DEFAULT_MODEL,
) -> bool:
    expected = assistant.faq_by_id[expected_id]
    response = assistant.client.chat(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict answer evaluator. Decide whether the candidate answer correctly "
                    "and faithfully answers the question using the expected FAQ answer, allowing a "
                    "concise translation into the customer's language. A refusal/NO_ANSWER is incorrect "
                    "for this labeled-answerable question. Output exactly YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nExpected FAQ answer: {expected.answer}\n"
                    f"Candidate answer: {answer}"
                ),
            },
        ],
        options={"temperature": 0},
    )
    result = ollama_content(response).strip().upper()
    if result not in {"YES", "NO"}:
        raise ValueError(f"Local correctness judge returned an invalid response: {result!r}")
    return result == "YES"


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    answerable = [row for row in rows if row["answerable"] == "yes"]
    unanswerable = [row for row in rows if row["answerable"] == "no"]
    return {
        "accuracy_answerable": sum(bool(row["correct"]) for row in answerable) / len(answerable)
        if answerable
        else math.nan,
        "idk_recall": sum(bool(row["is_idk"]) for row in unanswerable) / len(unanswerable)
        if unanswerable
        else math.nan,
        "false_idk_rate": sum(bool(row["is_idk"]) for row in answerable) / len(answerable)
        if answerable
        else math.nan,
        "language_match_rate": sum(row["language"] == row["predicted_language"] for row in rows)
        / len(rows)
        if rows
        else math.nan,
        "latency_median_seconds": statistics.median(row["latency_seconds"] for row in rows)
        if rows
        else math.nan,
        "latency_p95_seconds": percentile([row["latency_seconds"] for row in rows], 0.95),
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = load_inputs(args.questions, args.labels)
    assistant = BankAssistant(
        model=args.model,
        threshold=args.threshold,
        banglish_rewrite=args.banglish_rewrite,
    )
    output: list[dict[str, Any]] = []
    for item in data.itertuples(index=False):
        result = assistant.ask(item.question)
        denied = is_idk(result["answer"], item.language)
        reply_language = detect_reply_language(
            result["answer"], item.language, assistant.client, assistant.model
        )
        row = {
            "id": item.id,
            "question": item.question,
            "language": item.language,
            "answerable": item.answerable,
            "expected_faq_id": item.expected_faq_id,
            "answer": result["answer"],
            "detected_language": result["language"],
            "predicted_language": reply_language,
            "retrieved_ids": json.dumps(result["retrieved_ids"], ensure_ascii=False),
            "top_score": result["top_score"],
            "latency_seconds": result["latency_seconds"],
            "is_idk": denied,
            "correct": False,
            "correctness_method": f"local {args.judge_model} judge against expected FAQ answer",
        }
        output.append(row)
        print(f"[{len(output):02d}/40] {item.id}: {result['answer']}")
        pd.DataFrame(output).to_csv(args.output, index=False, encoding="utf-8-sig")

    for row in output:
        if row["answerable"] == "yes" and not row["is_idk"]:
            row["correct"] = judge_correctness(
                assistant,
                row["question"],
                row["expected_faq_id"],
                row["answer"],
                args.judge_model,
            )
            pd.DataFrame(output).to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\nMetrics (accuracy uses the local Ollama LLM judge):")
    for name, value in calculate_metrics(output).items():
        print(f"{name}: {value:.4f}")
    print(f"\nPer-question results saved to {args.output}")
    return output


def sweep_thresholds(args: argparse.Namespace) -> None:
    data = load_inputs(args.questions, args.labels)
    assistant = BankAssistant(model=args.model, threshold=args.threshold)
    scores: list[tuple[str, float]] = []
    for item in data.itertuples(index=False):
        _, score = assistant.threshold_prediction(item.question)
        scores.append((item.answerable, score))
    print("threshold,idk_recall,false_idk_rate")
    for threshold in [0.30 + 0.05 * step for step in range(8)]:
        predictions = [(label, score < threshold) for label, score in scores]
        unknown = [(label, denied) for label, denied in predictions if label == "no"]
        known = [(label, denied) for label, denied in predictions if label == "yes"]
        recall = sum(denied for _, denied in unknown) / len(unknown)
        false_rate = sum(denied for _, denied in known) / len(known)
        print(f"{threshold:.2f},{recall:.4f},{false_rate:.4f}")
    print("Threshold sweep is exploratory; this 40-question set is too small to avoid overfitting.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results_local.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--banglish-rewrite", action="store_true")
    parser.add_argument("--sweep-thresholds", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        if args.sweep_thresholds:
            sweep_thresholds(args)
        else:
            evaluate(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

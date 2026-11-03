"""Build the comparison table and local failure examples from saved evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluate import calculate_metrics

ROOT = Path(__file__).resolve().parent


def _read_result(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, keep_default_na=False)
    rows = frame.to_dict(orient="records")
    for row in rows:
        for field in ("correct", "is_idk"):
            value = row.get(field, False)
            row[field] = value is True or str(value).strip().lower() in {"true", "1", "yes"}
        row["latency_seconds"] = float(row["latency_seconds"])
    return rows


def _metric_line(name: str, rows: list[dict[str, Any]]) -> str:
    metrics = calculate_metrics(rows)
    return (
        f"| {name} | {metrics['accuracy_answerable']:.3f} | {metrics['idk_recall']:.3f} "
        f"| {metrics['false_idk_rate']:.3f} | {metrics['language_match_rate']:.3f} "
        f"| {metrics['latency_median_seconds']:.2f} | {metrics['latency_p95_seconds']:.2f} |"
    )


def _failure_reason(row: dict[str, Any]) -> str:
    if row["language"] != row["predicted_language"]:
        return "Reply language did not match the labeled question language."
    if row["answerable"] == "yes" and row["is_idk"]:
        return f"False IDK; the FAQ labels this as answerable (FAQ {row['expected_faq_id']})."
    if row["answerable"] == "yes" and not row["correct"]:
        return f"Answer did not pass the local judge against FAQ {row['expected_faq_id']}."
    if row["answerable"] == "no" and not row["is_idk"]:
        return "Answered a question labeled as outside the FAQ."
    return "Failed at least one evaluation check."


def write_comparison(
    local_path: Path = ROOT / "results_local.csv",
    online_path: Path = ROOT / "results_online.csv",
    report_path: Path = ROOT / "comparison.md",
    qwen_path: Path = ROOT / "results_qwen.csv",
    rewrite_path: Path = ROOT / "results_rewrite.csv",
) -> bool:
    if not local_path.is_file():
        return False
    local = _read_result(local_path)
    lines = [
        "# Evaluation comparison",
        "",
        "Metrics use the reviewed 40-question labels with a fixed Gemma3 12B local judge against "
        "the expected FAQ answer; it is not human verification. Retrieval threshold: 0.60 "
        "(selected on this 40-question set; threshold tuning may overfit).",
        "",
        "| Model | Accuracy (answerable) | IDK recall | False-IDK rate | Language match | Median latency (s) | P95 latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_line("Gemma3 12B (selected local model)", local),
    ]
    if qwen_path.is_file():
        lines.append(_metric_line("Qwen2.5 7B (local bake-off)", _read_result(qwen_path)))
    if rewrite_path.is_file():
        lines.append(
            _metric_line("Gemma3 12B + Banglish rewrite (opt-in)", _read_result(rewrite_path))
        )
    if online_path.is_file():
        lines.append(_metric_line("Gemini 3.5 Flash Lite (online)", _read_result(online_path)))
    else:
        lines.append(
            "| Gemini (online) | Skipped: fresh API key required | N/A | N/A | N/A | N/A | N/A |"
        )
    lines.extend([
        "| Fine-tuned local model | Not run (optional) | Not run | Not run | Not run | Not run | Not run |",
        "",
        "## Five local failure examples",
        "",
    ])
    failures = [
        row
        for row in local
        if row["language"] != row["predicted_language"]
        or (row["answerable"] == "yes" and (row["is_idk"] or not row["correct"]))
        or (row["answerable"] == "no" and not row["is_idk"])
    ]
    if failures:
        for row in failures[:5]:
            question = str(row["question"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"- **Question {row['id']}:** {question} — {_failure_reason(row)}")
    else:
        lines.append("No local failures were recorded; five failure examples are therefore unavailable.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The 40 questions are a small sample; threshold selection on them can overfit.",
            "- Labels were confirmed by the user as best-effort guesses, and the correctness judge can itself be wrong.",
            "- The online comparison uses the same retrieval code and prompt but adds network/API latency.",
            "- The online run sends these synthetic questions and retrieved FAQ text to the provider; never run it on real customer data.",
            "- Fine-tuning is optional and was not performed.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison report saved to {report_path}")
    return True

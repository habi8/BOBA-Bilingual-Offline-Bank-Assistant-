"""Write a paired before/after report for Qwen base versus QLoRA results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate import calculate_metrics

ROOT = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, keep_default_na=False)
    rows = frame.to_dict(orient="records")
    for row in rows:
        for field in ("correct", "is_idk"):
            value = row.get(field, False)
            row[field] = value is True or str(value).strip().lower() in {"true", "1", "yes"}
        row["latency_seconds"] = float(row["latency_seconds"])
    return rows


def metric_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics['accuracy_answerable']:.3f} "
        f"| {metrics['idk_recall']:.3f} | {metrics['false_idk_rate']:.3f} "
        f"| {metrics['language_match_rate']:.3f} "
        f"| {metrics['latency_median_seconds']:.2f} "
        f"| {metrics['latency_p95_seconds']:.2f} |"
    )


def write_report(before_path: Path, after_path: Path, output_path: Path) -> None:
    before = read_rows(before_path)
    after = read_rows(after_path)
    before_ids = [str(row["id"]) for row in before]
    after_ids = [str(row["id"]) for row in after]
    if len(before) != 40 or len(after) != 40 or before_ids != after_ids:
        raise ValueError("Before and after files must contain the same 40 questions in order.")

    before_metrics = calculate_metrics(before)
    after_metrics = calculate_metrics(after)
    lines = [
        "# QLoRA fine-tuning: before and after",
        "",
        "The before run uses the base Gemma3 12B Ollama model; the after run uses its QLoRA "
        "adapter. Both use the same reviewed labels, retrieval threshold, 40-question held-out "
        "set, and local Gemma3 12B correctness judge. Labels and LLM-judge decisions are not "
        "human-verified ground truth. Treat differences as directional on this small sample.",
        "",
        "| Model | Accuracy (answerable) | IDK recall | False-IDK rate | Language match | Median latency (s) | P95 latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_line("Base Gemma3 12B", before_metrics),
        metric_line("Gemma3 12B + QLoRA", after_metrics),
        "",
        "## Per-question changes",
        "",
    ]
    changes = [
        (old, new)
        for old, new in zip(before, after)
        if old["correct"] != new["correct"]
        or old["is_idk"] != new["is_idk"]
        or old["predicted_language"] != new["predicted_language"]
    ]
    if changes:
        for old, new in changes[:10]:
            question = str(new["question"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"- **Question {new['id']}:** {question} — "
                f"correct {old['correct']}→{new['correct']}, "
                f"IDK {old['is_idk']}→{new['is_idk']}, "
                f"language {old['predicted_language']}→{new['predicted_language']}."
            )
    else:
        lines.append("No differences were observed in correctness, abstention, or language.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Only 90 reviewed answerable examples were used for training; no refusal examples were included.",
            "- Fine-tuning may overfit these examples and does not improve FAQ retrieval itself.",
            "- The test set contains only 40 questions, and labels were best-effort guesses.",
            "- The QLoRA and Ollama base models use different quantization/runtime paths.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Before/after report saved to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "results_gemma_before.csv")
    parser.add_argument("--after", type=Path, default=ROOT / "results_finetuned.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "before_after.md")
    args = parser.parse_args()
    write_report(args.before, args.after, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

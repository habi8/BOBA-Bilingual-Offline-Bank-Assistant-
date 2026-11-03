"""Compare the same FAQ context and prompt through Google's Gemini API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values

from assistant import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    IDK,
    BankAssistant,
    answer_prompt,
    detect_language,
    detect_reply_language,
    enforce_output_language,
    is_idk,
)
from evaluate import calculate_metrics, judge_correctness, load_inputs
from report import write_comparison

ROOT = Path(__file__).resolve().parent


def api_key() -> str | None:
    accepted_names = {"gemini_api_key", "google_api_key"}
    for path in (ROOT / ".env.local", ROOT / ".env"):
        for name, value in dotenv_values(path).items():
            if name.lower() in accepted_names and value and value.strip():
                return value.strip()
    return None


def call_gemini(key: str, model: str, messages: list[dict[str, str]]) -> str:
    system = messages[0]["content"]
    user = messages[1]["content"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(key)}"
    )
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data: dict[str, Any] | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
            print(f"Gemini temporary HTTP {exc.code}; retrying in {2 ** (attempt + 1)} seconds.")
            time.sleep(2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(f"Gemini API request failed: {reason}") from exc
            print(f"Gemini request temporarily failed; retrying in {2 ** (attempt + 1)} seconds.")
            time.sleep(2 ** (attempt + 1))
    if data is None:
        raise RuntimeError("Gemini request ended without a response.")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini API response had no text: {data}") from exc


def evaluate_online(args: argparse.Namespace) -> None:
    key = api_key()
    if not key:
        print(
            "Online comparison skipped: add GEMINI_API_KEY=<key> to the project-root .env.local "
            "or .env file, then run evaluate_online.py again."
        )
        write_comparison(online_path=args.output)
        return
    data = load_inputs(args.questions, args.labels)
    assistant = BankAssistant(
        model=args.local_judge_model,
        threshold=args.threshold,
        banglish_rewrite=args.banglish_rewrite,
    )
    output: list[dict[str, Any]] = []
    for item in data.itertuples(index=False):
        started = time.perf_counter()
        language = detect_language(item.question, assistant.client, assistant.model)
        retrieved = assistant.retrieve(item.question, language)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        if not retrieved or top_score < args.threshold:
            answer = IDK[language]
        else:
            messages = answer_prompt(
                language, "\n\n".join(x["text"] for x in retrieved), item.question
            )
            answer = call_gemini(
                key,
                args.online_model,
                messages,
            )
            answer = enforce_output_language(
                answer,
                language,
                assistant.client,
                assistant.model,
                lambda previous_answer: call_gemini(
                    key,
                    args.online_model,
                    answer_prompt(
                        language,
                        "\n\n".join(x["text"] for x in retrieved),
                        item.question,
                        previous_answer,
                    ),
                ),
            )
        latency = time.perf_counter() - started
        denied = is_idk(answer, item.language)
        reply_language = detect_reply_language(answer, item.language, assistant.client, assistant.model)
        correct = False
        if item.answerable == "yes" and not denied:
            correct = judge_correctness(assistant, item.question, item.expected_faq_id, answer)
        output.append(
            {
                "id": item.id,
                "question": item.question,
                "language": item.language,
                "answerable": item.answerable,
                "expected_faq_id": item.expected_faq_id,
                "answer": answer,
                "detected_language": language,
                "predicted_language": reply_language,
                "retrieved_ids": json.dumps([x["faq_id"] for x in retrieved], ensure_ascii=False),
                "top_score": top_score,
                "latency_seconds": latency,
                "is_idk": denied,
                "correct": correct,
                "correctness_method": f"local {args.local_judge_model} judge against expected FAQ answer",
            }
        )
        pd.DataFrame(output).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[{len(output):02d}/40] {item.id}: {answer}")

    print("\nOnline metrics:")
    for name, value in calculate_metrics(output).items():
        print(f"{name}: {value:.4f}")
    print(f"\nPer-question results saved to {args.output}")
    write_comparison(online_path=args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "test-questions.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results_online.csv")
    parser.add_argument("--online-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--local-judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--banglish-rewrite", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        evaluate_online(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

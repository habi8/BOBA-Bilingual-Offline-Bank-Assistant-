"""Ask questions using the fine-tuned Gemma 3 12B FAQ assistant."""

from __future__ import annotations

import argparse
import random
import sys
import threading
from pathlib import Path

from assistant import DEFAULT_THRESHOLD, BankAssistant
from evaluate_finetuned import (
    DEFAULT_ADAPTER,
    DEFAULT_BASE_MODEL,
    TransformersChatClient,
)


def generate_welcome(client: TransformersChatClient, model: str) -> str:
    styles = (
        "warm and welcoming",
        "friendly and relaxed",
        "polite and professional",
        "cheerful but concise",
        "natural and conversational",
    )
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write one brief, fresh opening welcome for a bilingual bank FAQ "
                    "assistant. Use English. Greet the user naturally, invite them to ask "
                    "how you can help, and do not invent bank policies. Avoid a stock "
                    "'Hi! Hello!' opening. Vary the wording and sentence structure."
                ),
            },
            {"role": "user", "content": f"Use a {random.choice(styles)} tone."},
        ],
        options={"temperature": 0.85, "num_predict": 48, "top_p": 0.9},
    )
    return response["message"]["content"].strip()


def ask_with_thinking(assistant: BankAssistant, question: str) -> str:
    stop = threading.Event()
    dots = 0

    def animate() -> None:
        nonlocal dots
        while not stop.wait(0.45):
            dots = dots % 3 + 1
            sys.stdout.write(f"\rAssistant: {'.' * dots}{' ' * (3 - dots)}")
            sys.stdout.flush()

    sys.stdout.write("Assistant: .")
    sys.stdout.flush()
    animation = threading.Thread(target=animate, daemon=True)
    animation.start()
    try:
        answer = assistant.ask(question)["answer"]
    finally:
        stop.set()
        animation.join()
        sys.stdout.write("\r" + " " * 32 + "\r")
        sys.stdout.flush()
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="Optional one-off question; omit it for interactive mode",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--banglish-rewrite", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        client = TransformersChatClient(args.base_model, args.adapter)
        assistant = BankAssistant(
            model=args.base_model,
            threshold=args.threshold,
            banglish_rewrite=args.banglish_rewrite,
            client=client,
        )
        if args.question:
            print(ask_with_thinking(assistant, args.question))
            return 0

        print("Fine-tuned FAQ assistant ready. Type 'exit' or 'quit' to stop.")
        print(f"Assistant: {generate_welcome(client, args.base_model)}")
        while True:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            print(ask_with_thinking(assistant, question))
        return 0
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

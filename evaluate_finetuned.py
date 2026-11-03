"""Evaluate the trained Gemma 3 12B QLoRA adapter on the unchanged test set."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from assistant import (
    DEFAULT_THRESHOLD,
    BankAssistant,
    detect_reply_language,
    is_idk,
)
from evaluate import calculate_metrics, judge_correctness, load_inputs

ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER = ROOT / "output" / "gemma3-12b-bank-lora"
DEFAULT_OUTPUT = ROOT / "results_finetuned.csv"
DEFAULT_BASE_MODEL = "unsloth/gemma-3-12b-it-bnb-4bit"


class TransformersChatClient:
    def __init__(self, base_model: str, adapter_path: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load the 4-bit fine-tuned model.")
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No trained LoRA adapter found at {adapter_path}.")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        config = AutoConfig.from_pretrained(base_model)
        model_class = (
            AutoModelForImageTextToText
            if config.model_type == "gemma3"
            else AutoModelForCausalLM
        )
        base = model_class.from_pretrained(
            base_model,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.float16,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, str]]:
        del model
        generation_options = options or {}
        temperature = float(generation_options.get("temperature", 0))
        generation_args: dict[str, Any] = {
            "max_new_tokens": int(generation_options.get("num_predict", 160)),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
            if "top_p" in generation_options:
                generation_args["top_p"] = float(generation_options["top_p"])
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                **generation_args,
            )
        prompt_length = encoded["input_ids"].shape[1]
        content = self.tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        return {"message": {"content": content}}


def save_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = load_inputs(args.questions, args.labels)
    client = TransformersChatClient(args.base_model, args.adapter)
    assistant = BankAssistant(
        model=args.base_model,
        threshold=args.threshold,
        banglish_rewrite=args.banglish_rewrite,
        client=client,
    )
    rows: list[dict[str, Any]] = []
    for item in data.itertuples(index=False):
        result = assistant.ask(item.question)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "language": item.language,
                "answerable": item.answerable,
                "expected_faq_id": item.expected_faq_id,
                "answer": result["answer"],
                "detected_language": result["language"],
                "predicted_language": detect_reply_language(
                    result["answer"], item.language, client, args.base_model
                ),
                "retrieved_ids": json.dumps(result["retrieved_ids"], ensure_ascii=False),
                "top_score": result["top_score"],
                "latency_seconds": result["latency_seconds"],
                "is_idk": is_idk(result["answer"], item.language),
                "correct": False,
                "correctness_method": f"local {args.judge_model} judge against expected FAQ answer",
            }
        )
        save_rows(rows, args.output)
        print(f"[{len(rows):02d}/40] {item.id}: {result['answer']}")

    del assistant
    del client
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    judge = BankAssistant(model=args.judge_model)
    for row in rows:
        if row["answerable"] == "yes" and not row["is_idk"]:
            row["correct"] = judge_correctness(
                judge,
                row["question"],
                row["expected_faq_id"],
                row["answer"],
                args.judge_model,
            )
            save_rows(rows, args.output)

    print("\nFine-tuned metrics (correctness uses the fixed local Ollama judge):")
    for name, value in calculate_metrics(rows).items():
        print(f"{name}: {value:.4f}")
    print(f"\nPer-question results saved to {args.output}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "test-questions.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels.csv")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--judge-model", default="gemma3:12b")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--banglish-rewrite", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        evaluate(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

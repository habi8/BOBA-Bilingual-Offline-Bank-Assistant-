"""QLoRA fine-tune Gemma 3 12B on reviewed, FAQ-grounded examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "finetune_examples.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "gemma3-12b-bank-lora"
DEFAULT_BASE_MODEL = "unsloth/gemma-3-12b-it-bnb-4bit"


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("review_status") != "reviewed":
                raise ValueError(f"{path}:{line_number} is not marked reviewed.")
            if not all(row.get(field) for field in ("faq_id", "language", "question", "context", "answer")):
                raise ValueError(f"{path}:{line_number} is missing a required field.")
            examples.append(row)
    if not examples:
        raise ValueError(f"No reviewed examples found in {path}.")
    return examples


def encode_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    system = (
        "Answer only from the provided bank FAQ context. Do not add or infer facts. "
        f"Reply in {row['language']}. If the context does not answer the question, abstain."
    )
    user = f"CONTEXT:\n{row['context']}\n\nQUESTION:\n{row['question']}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["answer"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )["input_ids"]
    if len(prompt_ids) >= len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template produced an invalid prompt/answer boundary.")
    if len(full_ids) > max_length:
        raise ValueError(
            f"Example FAQ {row['faq_id']} needs {len(full_ids)} tokens; "
            f"the configured maximum is {max_length}. Increase --max-length."
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids, masks, labels = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements-finetune.txt "
            "and the CUDA-enabled PyTorch build described in README.md."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install CUDA-enabled PyTorch "
            "and verify torch.cuda.is_available() before training; CPU-only QLoRA is not supported."
        )
    free_vram, _ = torch.cuda.mem_get_info()
    if free_vram < 8 * 1024**3:
        raise RuntimeError(
            f"Only {free_vram / 1024**3:.1f} GiB GPU memory is free; need at least 8 GiB "
            "for this QLoRA configuration. Close GPU-heavy applications and retry."
        )

    examples = load_examples(args.data)
    print(f"Loading {len(examples)} reviewed examples from {args.data}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Free VRAM: {free_vram / 1024**3:.1f} GiB")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    config = AutoConfig.from_pretrained(args.base_model)
    model_class = (
        AutoModelForImageTextToText
        if config.model_type == "gemma3"
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    adapter = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()

    encoded = [
        encode_example(tokenizer, row, args.max_length)
        for row in examples
    ]
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=3,
        lr_scheduler_type="cosine",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "base_model": args.base_model,
        "method": "QLoRA NF4",
        "epochs": args.epochs,
        "max_length": args.max_length,
        "training_examples": len(examples),
        "held_out_evaluation": "test-questions.csv (not used for training)",
    }
    (args.output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Adapter saved to {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    try:
        train(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

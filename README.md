# Bilingual Offline Bank Assistant

The project has two local assistant paths:

- **Fine-tuned assistant (recommended interactive workflow):** `ask_finetuned.py` loads the Gemma 3 12B QLoRA adapter through Transformers and PEFT. A local model classifies each message as banking-related or out of scope and detects English, Bangla, or Banglish. Banking requests use retrieval from `bank-faq.csv`; out-of-scope messages receive a brief notice. The interactive CLI displays a fixed greeting, shows animated dots while answering, and keeps the model loaded for the session.
- **Ollama baseline:** `assistant.py` and `evaluate.py` use the regular Ollama model, not the QLoRA adapter. Use these for the base-model comparison and original evaluation workflow.

Banking answers must be grounded in the FAQ. After retrieval, the assistant checks that the FAQ directly answers the question; it abstains in the detected input language when the FAQ is merely related, covers a different product, or lacks the requested private account information. Questions unrelated to the bank FAQ are not answered.

## Run the fine-tuned assistant

Run these commands in PowerShell from the project directory:

```powershell
# Uses the default Hugging Face base model and output\gemma3-12b-bank-lora adapter.
.\.venv\Scripts\python.exe .\ask_finetuned.py

# Or point to a base checkpoint already downloaded locally:
.\.venv\Scripts\python.exe .\ask_finetuned.py --base-model "E:\Models\gemma-3-12b-it-bnb-4bit"
```

Wait for the generated welcome, then type messages at the `You:` prompt. Type `exit` or `quit` to end the session. The adapter is loaded once and reused for each message.

The command above assumes:

- The local base checkpoint is at `E:\Models\gemma-3-12b-it-bnb-4bit`.
- The trained adapter is at `output\gemma3-12b-bank-lora`.
- CUDA-enabled PyTorch, Transformers, PEFT, and bitsandbytes are installed.
- A CUDA-capable GPU with sufficient free VRAM is available (training/evaluation used an RTX 3060).

The default base model is `unsloth/gemma-3-12b-it-bnb-4bit`; omit `--base-model` to load it from Hugging Face. To ask one question and exit:

```powershell
.\.venv\Scripts\python.exe .\ask_finetuned.py "What documents do I need to open a savings account?"
```

`assistant.py` does **not** load the adapter. Run `ask_finetuned.py` to use the fine-tuned model.

## Setup and model files

1. Install Python 3.10+ and NVIDIA drivers for CUDA.
2. Create and activate the environment:
   - Windows: `py -3.10 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install the base project dependencies: `python -m pip install -r requirements.txt`.
4. For the fine-tuned assistant, install CUDA PyTorch and fine-tuning/runtime dependencies. On the Windows training environment, the tested commands were:

   ```powershell
   python -m pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
   python -m pip install -r requirements-finetune.txt
   ```

5. The base checkpoint is the public `unsloth/gemma-3-12b-it-bnb-4bit` model (about 7.3 GB); the adapter is stored in `output/gemma3-12b-bank-lora`. If loading from Hugging Face fails or stalls, download the checkpoint to a drive with enough free space. This resumable PowerShell example uses `curl.exe`:

   ```powershell
   $dir = "E:\Models\gemma-3-12b-it-bnb-4bit"
   $repo = "https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit/resolve/main"
   New-Item -ItemType Directory -Path $dir -Force | Out-Null
   $files = @("config.json", "generation_config.json", "model.safetensors.index.json",
       "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
       "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
       "added_tokens.json", "chat_template.jinja", "chat_template.json",
       "preprocessor_config.json", "processor_config.json", "tokenizer.model")
   foreach ($file in $files) {
       curl.exe -L --fail --retry 8 --retry-all-errors --retry-delay 3 -C - `
           -o (Join-Path $dir $file) "$repo/$file"
       if ($LASTEXITCODE -ne 0) { throw "Download failed: $file" }
   }
   ```

   If interrupted, rerun the loop with the same destination; curl resumes partial files.

## Assistant behavior

- The local model classifies message intent; this is not a fixed list of greeting phrases. Banking questions (including a greeting followed by a banking question) go through FAQ retrieval. Non-banking requests receive a brief scope notice and are not answered conversationally.
- The interactive session starts with the fixed greeting “Hello! How can I help you today?” Replies to user messages should match the detected input language: English, Bangla script, or Banglish (Bengali written with Latin characters).
- For Banglish, the response should use Bengali words/grammar written in Latin script, not Bengali script or ordinary English. The language check retries a mismatched reply up to three times, then returns a localized fallback if it still cannot produce a matching-language answer.
- `bank-faq.csv` is the only source for bank-specific answers. The retrieval threshold defaults to `0.55`; lower scores return an “I don't know” response in the input language. Optional Banglish-to-English retrieval rewriting is disabled unless `--banglish-rewrite` is specified.
- The local FAQ index uses BGE-M3 embeddings and FAISS. Its cache is invalidated when the FAQ file or embedding model changes.
- The fine-tuning set contains 90 reviewed answerable examples: one example for each FAQ in English, Bangla, and Banglish. It does not contain refusal examples, so fine-tuning is not intended to teach abstention. Do not treat the adapter as safer than the baseline.

## Training and evaluation

- Rebuild/validate the approved examples with `python prepare_finetune_data.py`. The 40-question evaluation set is intentionally excluded from training.
- Train the adapter with `python finetune.py --base-model "E:\Models\gemma-3-12b-it-bnb-4bit" --max-length 512`. Training uses QLoRA with answer-only loss and writes the adapter to `output/gemma3-12b-bank-lora`.
- Evaluate the adapter on the 40 questions with:

  ```powershell
  python evaluate_finetuned.py --base-model "E:\Models\gemma-3-12b-it-bnb-4bit" --output results_finetuned.csv
  ```

  This also requires Ollama with `gemma3:12b` available for the local correctness judge. The adapter runs through Transformers/bitsandbytes; the judge runs after the adapter is released from GPU memory.
- For a paired before/after report, first run `python evaluate.py --model gemma3:12b --threshold 0.55 --output results_gemma_before.csv`, then run the fine-tuned evaluation above, and finally run `python report_finetune.py`. The report is written to `before_after.md`.
- The latest recorded comparison is in `before_after.md`; per-question results are in `results_gemma_before.csv` and `results_finetuned.csv`. It is a small, automatically judged sample; inspect failures and do not claim generalization from these 40 questions.
- Evaluation labels in `labels.csv` are best-effort guesses, not human-verified ground truth. Correctness is assessed by a local LLM judge, not a human reviewer.

## Ollama baseline and optional online comparison

- Install [Ollama](https://ollama.com), then run `python setup_models.py` to check/download the baseline models `qwen2.5:7b-instruct`, `gemma3:12b`, and the BGE-M3 embeddings. The script may download several gigabytes.
- Ask a one-off baseline question with `python assistant.py "What is the debit card annual fee?"`.
- Evaluate the baseline with `python evaluate.py`; opt-in Banglish query rewriting with `python evaluate.py --banglish-rewrite`. The threshold sweep is `python evaluate.py --sweep-thresholds`.
- For an online comparison only, add `GEMINI_API_KEY=...` to a root `.env.local` or `.env` file and run `python evaluate_online.py`. The evaluator prefers `.env.local`. Never commit either environment file. This sends synthetic test questions and retrieved FAQ context to Gemini; do not use it with real customer data.
- `results_local.csv`, `results_online.csv`, and generated comparison reports contain synthetic questions and model responses.

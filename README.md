# Bilingual Offline Bank Assistant

Banking answers are grounded in `bank-faq.csv`; a local LLM classifies whether a message needs bank FAQ retrieval or a normal conversational reply, rather than relying on a fixed list of greeting phrases. Question classification, retrieval, generation, and correctness judging run locally through Ollama; only the explicitly requested online comparison sends the synthetic test questions and retrieved FAQ context to Gemini.

## Run order

1. Install Python 3.10+ and [Ollama](https://ollama.com); install NVIDIA drivers if using a supported GPU.
2. Create and activate the environment:
   - Windows: `py -3.10 -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install packages: `python -m pip install -r requirements.txt`
4. Review the proposed labels in `labels.csv`. They are best-effort guesses, not ground truth. Do not treat evaluation metrics as final until they are reviewed and confirmed.
5. Run `python setup_models.py` to check/download Qwen2.5 7B, Gemma3 12B, and BGE-M3.
6. Ask a local question: `python assistant.py "What is the debit card annual fee?"`
7. Evaluate the selected Gemma model: `python evaluate.py`; to test Banglish query rewriting use `python evaluate.py --banglish-rewrite`.
8. The Qwen-vs-Gemma bake-off has been run; Qwen can be re-evaluated with `python evaluate.py --model qwen2.5:7b-instruct --output results_qwen.csv`.
9. For online comparison, add a fresh `GEMINI_API_KEY=...` to a root `.env.local` or `.env` file and run `python evaluate_online.py`. The evaluator prefers `.env.local` and currently uses `gemini-3.5-flash-lite`. Never commit either env file.

## Fine-tuning data preparation

- Run `python prepare_finetune_data.py` to create `finetune_examples.jsonl`: one FAQ-grounded example per FAQ in English, Bangla, and Banglish (90 total). Each includes the source FAQ context. The user reviewed and approved these examples; each is marked `review_status: reviewed`. The builder intentionally excludes the 40 evaluation questions.
- QLoRA trainer: on this Windows/Python 3.10 environment with the RTX 3060, install CUDA PyTorch with `python -m pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`, then run `python -m pip install -r requirements-finetune.txt` and `python finetune.py`. This uses the public 4-bit `unsloth/gemma-3-12b-it-bnb-4bit` checkpoint (about 7.3 GB) and requires at least 8 GiB free GPU memory. It saves a LoRA adapter under `output/gemma3-12b-bank-lora`. It trains on answer-only tokens, leaving the FAQ context and question out of the loss.
- If Hugging Face's large-file transfer stalls, download directly to a drive with enough free space. In PowerShell, this resumable `curl.exe -C -` sequence downloads the Gemma-specific files (do not request Qwen's `vocab.json` or `merges.txt`):

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
  python finetune.py --base-model $dir --max-length 512
  ```

  If interrupted, rerun the loop with the same destination; curl resumes partial files.
- After training, run `python evaluate_finetuned.py` to evaluate the adapter against the unchanged 40 questions, using the same retrieval threshold and fixed Gemma3 12B local correctness judge. The adapter is loaded through Transformers/bitsandbytes; the judge runs only after the adapter is released from GPU memory.
- To chat with the adapter interactively, run `python ask_finetuned.py --base-model "E:\Models\gemma-3-12b-it-bnb-4bit"` and enter questions one at a time; type `exit` or `quit` to stop. The model loads once for the session, generates a varied English welcome, and shows animated dots while preparing each reply. A local LLM decides whether a message is ordinary conversation or needs bank FAQ retrieval, without matching a fixed list of greeting words. Language detection distinguishes English, Bangla, and Banglish so replies can match the input language. Bank-related questions still use FAQ retrieval and generation. You can also pass a question as an argument for a one-off run. This uses the fine-tuned adapter and FAQ retrieval pipeline without running the 40-question evaluation or correctness judge. Omit `--base-model` only if the base checkpoint is available from Hugging Face.
- For a paired report, first run `python evaluate.py --model gemma3:12b --threshold 0.55 --output results_gemma_before.csv`, then run `python evaluate_finetuned.py`, then `python report_finetune.py`. This writes `before_after.md`.
- The set contains answerable examples only. Add carefully reviewed out-of-FAQ refusal examples separately before any fine-tune intended to teach abstention; do not make up policy facts.

## Implementation and evaluation notes

- The FAQ CSV is parsed into one chunk per Q&A row. Its SHA-256 hash invalidates the normalized BGE-M3 embeddings and FAISS inner-product index only when the FAQ changes.
- Embeddings run on CPU to leave the RTX 3060's VRAM for Ollama. Ollama generation uses temperature 0. The retrieval threshold defaults to 0.55. This was lowered to recover more paraphrased questions, but may admit weaker FAQ matches and should be validated beyond the 40-question sample. `python evaluate.py --sweep-thresholds` prints the requested 0.30–0.65 recall/false-IDK sweep.
- Bangla is detected by Unicode script; English versus Banglish is classified by the local Ollama model. Optional Banglish-to-English retrieval rewriting is disabled by default.
- Answer correctness is judged by a fixed Gemma3 12B local judge against the expected FAQ answer in `labels.csv` for every compared model. This is an automated judge, not human verification; inspect the per-question outputs and failures.
- Language-match scoring classifies the generated reply itself (rather than only checking the input-language detector), so the output language is evaluated.
- The assistant checks each generated reply against the detected input language, retries a translation up to three times when they differ, and returns the localized “I don't know” response only if it cannot produce a reply that passes the language check.
- Latency measures question classification, retrieval, and response generation; correctness judging runs after all answers and is excluded from response latency.
- Threshold selection and evaluation use the same 40-question set, so a tuned threshold can overfit. Do not claim generalization from this small set.
- `results_local.csv` is the selected Gemma3 12B run; `results_qwen.csv` records the Qwen2.5 bake-off; `results_rewrite.csv` records the opt-in Banglish rewrite pass; `results_online.csv` records Gemini. The online evaluator generates `comparison.md` with the available results and local failure examples, marking Gemini as skipped when the key is absent.
- `results_local.csv`, `results_online.csv`, and any generated comparison report contain synthetic sample questions/responses. The online evaluator intentionally sends these samples plus retrieved FAQ text to Gemini; do not use it with real customer data.
- Fine-tuning is not run here. Prepare and review training data first, then compare any LoRA/QLoRA model against the unchanged 40-question evaluation set.

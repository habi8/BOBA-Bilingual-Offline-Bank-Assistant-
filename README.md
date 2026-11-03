# Bilingual Offline Bank Assistant

The assistant answers from `bank-faq.csv` only. Question classification, retrieval, generation, and correctness judging run locally through Ollama; only the explicitly requested online comparison sends the synthetic test questions and retrieved FAQ context to Gemini.

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

## Implementation and evaluation notes

- The FAQ CSV is parsed into one chunk per Q&A row. Its SHA-256 hash invalidates the normalized BGE-M3 embeddings and FAISS inner-product index only when the FAQ changes.
- Embeddings run on CPU to leave the RTX 3060's VRAM for Ollama. Ollama generation uses temperature 0. Threshold defaults to 0.60, selected from the 40-question sweep for full IDK recall with the lowest observed false-IDK rate at that recall; this is susceptible to overfitting. `python evaluate.py --sweep-thresholds` prints the requested 0.30–0.65 recall/false-IDK sweep.
- Bangla is detected by Unicode script; English versus Banglish is classified by the local Ollama model. Optional Banglish-to-English retrieval rewriting is disabled by default.
- Answer correctness is judged by a fixed Gemma3 12B local judge against the expected FAQ answer in `labels.csv` for every compared model. This is an automated judge, not human verification; inspect the per-question outputs and failures.
- Language-match scoring classifies the generated reply itself (rather than only checking the input-language detector), so the output language is evaluated.
- Latency measures question classification, retrieval, and response generation; correctness judging runs after all answers and is excluded from response latency.
- Threshold selection and evaluation use the same 40-question set, so a tuned threshold can overfit. Do not claim generalization from this small set.
- `results_local.csv` is the selected Gemma3 12B run; `results_qwen.csv` records the Qwen2.5 bake-off; `results_rewrite.csv` records the opt-in Banglish rewrite pass; `results_online.csv` records Gemini. The online evaluator generates `comparison.md` with the available results and local failure examples, marking Gemini as skipped when the key is absent.
- `results_local.csv`, `results_online.csv`, and any generated comparison report contain synthetic sample questions/responses. The online evaluator intentionally sends these samples plus retrieved FAQ text to Gemini; do not use it with real customer data.
- Fine-tuning is not included: it is optional and should only be attempted after the base workflow and reviewed-label evaluation work.

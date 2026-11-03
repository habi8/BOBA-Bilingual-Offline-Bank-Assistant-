# Evaluation comparison

Metrics use the reviewed 40-question labels with a fixed Gemma3 12B local judge against the expected FAQ answer; it is not human verification. Retrieval threshold: 0.60 (selected on this 40-question set; threshold tuning may overfit).

| Model | Accuracy (answerable) | IDK recall | False-IDK rate | Language match | Median latency (s) | P95 latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Gemma3 12B (selected local model) | 0.833 | 1.000 | 0.133 | 0.900 | 1.62 | 2.40 |
| Qwen2.5 7B (local bake-off) | 0.767 | 1.000 | 0.167 | 0.925 | 0.85 | 3.36 |
| Gemma3 12B + Banglish rewrite (opt-in) | 0.967 | 1.000 | 0.000 | 0.825 | 1.76 | 3.26 |
| Gemini 3.5 Flash Lite (online) | 0.867 | 1.000 | 0.133 | 1.000 | 1.19 | 2.65 |
| Fine-tuned local model | Not run (optional) | Not run | Not run | Not run | Not run | Not run |

## Five local failure examples

- **Question 1:** Personal loan er interest rate koto? — Reply language did not match the labeled question language.
- **Question 3:** Amar account theke ekta transaction ami korini, kothay janabo? — False IDK; the FAQ labels this as answerable (FAQ 24).
- **Question 7:** Ek din e koto taka transfer kora jay? — False IDK; the FAQ labels this as answerable (FAQ 7).
- **Question 23:** Debit card er annual fee koto? — Reply language did not match the labeled question language.
- **Question 24:** Card er PIN bhule gechi, reset korbo kibhabe? — Reply language did not match the labeled question language.

## Limitations

- The 40 questions are a small sample; threshold selection on them can overfit.
- Labels were confirmed by the user as best-effort guesses, and the correctness judge can itself be wrong.
- The online comparison uses the same retrieval code and prompt but adds network/API latency.
- The online run sends these synthetic questions and retrieved FAQ text to the provider; never run it on real customer data.
- Fine-tuning is optional and was not performed.

# QLoRA fine-tuning: before and after

The before run uses the base Gemma3 12B Ollama model; the after run uses its QLoRA adapter. Both use the same reviewed labels, retrieval threshold, 40-question held-out set, and local Gemma3 12B correctness judge. Labels and LLM-judge decisions are not human-verified ground truth. Treat differences as directional on this small sample.

| Model | Accuracy (answerable) | IDK recall | False-IDK rate | Language match | Median latency (s) | P95 latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Base Gemma3 12B | 0.833 | 1.000 | 0.133 | 1.000 | 1.74 | 4.99 |
| Gemma3 12B + QLoRA | 0.900 | 0.900 | 0.100 | 1.000 | 6.67 | 12.62 |

## Per-question changes

- **Question 13:** আমার বন্ধুর অ্যাকাউন্টে কত টাকা আছে বলুন তো? — correct False→False, IDK True→False, language Bangla→Bangla.
- **Question 24:** Card er PIN bhule gechi, reset korbo kibhabe? — correct False→True, IDK True→False, language Banglish→Banglish.
- **Question 31:** Credit card nite hole income koto hote hobe? — correct False→True, IDK False→False, language Banglish→Banglish.

## Limitations

- Only 90 reviewed answerable examples were used for training; no refusal examples were included.
- Fine-tuning may overfit these examples and does not improve FAQ retrieval itself.
- The test set contains only 40 questions, and labels were best-effort guesses.
- The QLoRA and Ollama base models use different quantization/runtime paths.

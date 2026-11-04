# Task 2 Loom Script

**Target length:** about 4 minutes 30 seconds  
**Recording note:** The score figures below are from the saved 40-question evaluation runs. They were recorded before the latest FAQ-relevance guard was added. Rerun the evaluations on the current code and replace the figures before recording, or say clearly that they are earlier results.

## 0:00–0:30 — Introduce the problem

**[Show the assistant and the sample FAQ file.]**

Hi, I’m Habib. I have compoleted task 2 which was a Bilingual Offline Bank assistant for customers who ask questions in English, Bangla, or Banglish.

The main challenge is to answer only from the bank’s approved FAQ. If the FAQ does not contain the answer, the assistant should not guess. It should say it doesn’t know or explain that the question is outside its scope.

## 0:30–1:20 — Show it working

**[Run the fine-tuned assistant. Ask one question in each language.]**

Here I’m asking an English question: “What is the late payment fee on a credit card?” The assistant finds the matching FAQ and answers with the listed fee.

Now I’ll ask in Bangla, then Banglish. The answer should follow the language I used, while still relying on the same bank FAQ.

**[Ask a question the FAQ cannot answer, such as whether the bank offers student loans. Then show an unrelated question or a request for another person’s account balance.]**

These are important tests. A personal-loan answer is not automatically an answer about student loans, and a public FAQ cannot tell us someone’s private account balance. The assistant should abstain instead of making a related FAQ look like a real answer.

## 1:20–2:10 — Explain how it works, without jargon

**[Show the FAQ retrieval flow or the relevant README section.]**

You can think of the FAQ as the assistant’s only approved reference book. When a question comes in, the system looks for the closest FAQ entry, checks whether it is relevant enough, and then asks the model to answer using that information only.

There are two safeguards. First, if the FAQ match is weak, it abstains. Second, a relevance check is meant to catch cases where a result is related but does not actually answer the question—for example, confusing student loans with personal loans.

The model also detects which of the three languages the customer used, so the reply can match. For Banglish, the project includes an optional step that rewrites the question in English to help find an English FAQ entry.

## 2:10–2:55 — Explain the fine-tuning

**[Show the training examples and the adapter folder, not the model weights.]**

As an optional improvement, I fine-tuned Gemma 3 12B using QLoRA. In plain language, instead of retraining the whole model, QLoRA teaches a small set of additional parameters while keeping the main model mostly unchanged. I used a four-bit model to reduce memory use.

The training set has 90 reviewed examples: one example for each of the 30 FAQ entries in English, Bangla, and Banglish. The 40 test questions were kept out of training. This teaches the model how to phrase FAQ answers in the different languages, but it does not fix a bad FAQ match by itself.

## 2:55–3:45 — Present the results honestly

**[Show the results table or comparison report.]**

On the saved 40-question test, the local base model correctly answered 83 percent of questions labelled answerable. After fine-tuning, that rose to 90 percent. However, the fine-tuned version correctly abstained on 90 percent of the questions labelled unanswerable, compared with 100 percent for the base run. Its median response time also increased from about 1.7 seconds to about 6.7 seconds.

For the separate online comparison, Gemini scored about 87 percent on answerable questions, abstained on all labelled unanswerable questions, and had a median response time of about 1.2 seconds.

These are small-sample results, not a guarantee of performance. The labels are best-effort, and an AI model—not a human—judged whether answers were correct. The fine-tuning and online comparison figures also come from separate runs, so I would not treat small differences between them as a definitive ranking.

## 3:45–4:30 — Limitations and next steps

**[Show the project README and, if available, the current evaluation output.]**

The hardest issue I did not fully solve is reliable matching for Banglish. A customer may write a question in Latin letters while the FAQ is in English, and a similar-looking result is not always the right answer. Rewriting the question can help, but it adds another model step and can still fail.

There are other limitations too. The FAQ and test set are small, the evaluation labels are not verified ground truth, and fine-tuning made responses slower in my recorded run. Although the assistant runs locally and does not need to send customer questions to an online AI service during normal use, the 12-billion-parameter model needs a capable computer and GPU; it is not suitable for every device. The separate online benchmark used synthetic sample questions and FAQ text, not real customer data.

My next steps would be to rerun all 40 questions against the latest version, review every failure with a person, add more reviewed Banglish and refusal examples, and test with a larger independent set. I would also measure performance on the intended bank hardware before recommending deployment.

That’s the solution: a local, multilingual FAQ assistant, with clear abstention behavior and an honest view of what still needs work. Thank you.

## Before recording

- Replace the saved metrics above if rerunning evaluation on the latest code; the current FAQ-relevance guard was added after those saved runs.
- Confirm each live demo question and answer on the machine used for the recording.
- Show the non-banking scope notice separately from the “I don’t know” answer for an in-scope banking question missing from the FAQ.
- Keep the sample data synthetic; never put real customer or account information into the online comparison.

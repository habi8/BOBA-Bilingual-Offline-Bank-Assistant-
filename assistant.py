"""Offline FAQ retrieval and response pipeline backed by Ollama."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "bank-faq.csv"
INDEX_DIR = ROOT / "cache"
INDEX_PATH = INDEX_DIR / "faq.index"
METADATA_PATH = INDEX_DIR / "faq-index.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_THRESHOLD = 0.60
IDK = {
    "English": "I don't know, let me connect you to a person.",
    "Bangla": "আমি জানি না, আপনাকে একজন প্রতিনিধির সঙ্গে কথা বলিয়ে দিচ্ছি।",
    "Banglish": "Ami jani na, apnake ekjon protinidhir sathe kotha boliye dichhi.",
}
NO_ANSWER = "NO_ANSWER"


@dataclass(frozen=True)
class FAQ:
    faq_id: str
    topic: str
    question: str
    answer: str

    @property
    def text(self) -> str:
        return f"FAQ ID: {self.faq_id}\nTopic: {self.topic}\nQuestion: {self.question}\nAnswer: {self.answer}"


def ollama_content(response: Any) -> str:
    message = response.get("message", {}) if isinstance(response, dict) else response.message
    content = message.get("content", "") if isinstance(message, dict) else message.content
    return str(content).strip()


def detect_language(question: str, client: Any, model: str) -> str:
    if re.search(r"[\u0980-\u09FF]", question):
        return "Bangla"
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the text's primary language as exactly one word: English or Banglish. "
                    "Banglish is Bangla written with Latin letters, for example 'Amar account e "
                    "taka ache?' or 'FDR korte minimum koto taka lage?'. Classify ordinary English "
                    "sentences as English, even if they mention Bangladesh or banking. Output no "
                    "explanation."
                ),
            },
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},
    )
    answer = ollama_content(response).lower()
    if "banglish" in answer:
        return "Banglish"
    if "english" in answer:
        return "English"
    raise ValueError(f"Language classifier returned an unexpected label: {answer!r}")


def detect_reply_language(answer: str, target_language: str, client: Any, model: str) -> str:
    if re.search(r"[\u0980-\u09FF]", answer):
        return "Bangla"
    text = re.sub(r"\b(?:BDT|USD|EUR|GBP|FDR|NID|ATM|OTP|TK|TAKA)\b", "", answer, flags=re.I)
    if not any(character.isalpha() for character in text):
        return target_language
    return detect_language(answer, client, model)


def answer_prompt(language: str, context: str, question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer only using the provided bank FAQ CONTEXT. Do not use outside knowledge or "
                "infer missing policies. If the context does not answer the question, output exactly "
                f"{NO_ANSWER}. Otherwise, give a short answer in {language}. For Bangla, use Bangla "
                "script. For Banglish, write Bangla words using Latin letters; do not use Bangla "
                "script or answer in English. For English, write standard English. Do not add a "
                "greeting or unsupported details."
            ),
        },
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
    ]


class BankAssistant:
    def __init__(
        self,
        *,
        faq_path: Path = FAQ_PATH,
        model: str = DEFAULT_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        banglish_rewrite: bool = False,
        cache_dir: Path = INDEX_DIR,
        client: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.faq_path = Path(faq_path)
        self.model = model
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.banglish_rewrite = banglish_rewrite
        self.client = client or ollama.Client()
        self.faqs = self._load_faqs()
        self.embedder = embedder or SentenceTransformer(embedding_model, device="cpu")
        self.index, self.faq_ids = self._load_or_build_index(Path(cache_dir))
        self.faq_by_id = {faq.faq_id: faq for faq in self.faqs}

    def _load_faqs(self) -> list[FAQ]:
        if not self.faq_path.is_file():
            raise FileNotFoundError(f"FAQ file not found: {self.faq_path}")
        frame = pd.read_csv(self.faq_path, dtype={"id": str}).fillna("")
        required = {"id", "topic", "question", "answer"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"FAQ CSV is missing required columns: {', '.join(sorted(missing))}")
        faqs = [
            FAQ(str(row.id), str(row.topic), str(row.question), str(row.answer))
            for row in frame.itertuples(index=False)
        ]
        if not faqs or len({faq.faq_id for faq in faqs}) != len(faqs):
            raise ValueError("FAQ must contain at least one row and unique IDs.")
        return faqs

    def _load_or_build_index(self, cache_dir: Path) -> tuple[faiss.Index, list[str]]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(self.faq_path.read_bytes()).hexdigest()
        metadata_path = cache_dir / METADATA_PATH.name
        index_path = cache_dir / INDEX_PATH.name
        if metadata_path.is_file() and index_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("faq_sha256") == digest and metadata.get("embedding_model") == self.embedding_model:
                return faiss.read_index(str(index_path)), [str(item) for item in metadata["faq_ids"]]

        vectors = self.embedder.encode(
            [faq.text for faq in self.faqs],
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.faqs):
            raise ValueError("Embedding model returned an invalid FAQ embedding matrix.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(
            json.dumps(
                {
                    "faq_sha256": digest,
                    "embedding_model": self.embedding_model,
                    "faq_ids": [faq.faq_id for faq in self.faqs],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return index, [faq.faq_id for faq in self.faqs]

    def _embed(self, text: str) -> np.ndarray:
        vector = self.embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vector, dtype=np.float32)

    def _rewrite_banglish(self, question: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user's Banglish banking question into concise English. "
                        "Preserve its meaning and output only the translation."
                    ),
                },
                {"role": "user", "content": question},
            ],
            options={"temperature": 0},
        )
        return ollama_content(response)

    def retrieve(self, question: str, language: str, top_k: int = 3) -> list[dict[str, Any]]:
        queries = [question]
        if language == "Banglish" and self.banglish_rewrite:
            rewritten = self._rewrite_banglish(question)
            if rewritten:
                queries.append(rewritten)

        candidates: dict[str, float] = {}
        for query in queries:
            scores, positions = self.index.search(self._embed(query), min(top_k, len(self.faqs)))
            for score, position in zip(scores[0], positions[0]):
                if position >= 0:
                    faq_id = self.faq_ids[int(position)]
                    candidates[faq_id] = max(float(score), candidates.get(faq_id, -1.0))
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {"faq_id": faq_id, "score": score, "text": self.faq_by_id[faq_id].text}
            for faq_id, score in ranked
        ]

    def answer_with_context(self, question: str, language: str, retrieved: list[dict[str, Any]]) -> str:
        context = "\n\n".join(item["text"] for item in retrieved)
        response = self.client.chat(
            model=self.model,
            messages=answer_prompt(language, context, question),
            options={"temperature": 0},
        )
        return ollama_content(response)

    def ask(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        language = detect_language(question, self.client, self.model)
        retrieved = self.retrieve(question, language)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        if not retrieved or top_score < self.threshold:
            answer = IDK[language]
        else:
            answer = self.answer_with_context(question, language, retrieved)
            if answer == NO_ANSWER:
                answer = IDK[language]
        return {
            "question": question,
            "answer": answer,
            "language": language,
            "retrieved_ids": [item["faq_id"] for item in retrieved],
            "top_score": top_score,
            "latency_seconds": time.perf_counter() - started,
        }

    def threshold_prediction(self, question: str) -> tuple[str, float]:
        language = detect_language(question, self.client, self.model)
        retrieved = self.retrieve(question, language)
        score = retrieved[0]["score"] if retrieved else 0.0
        return ("idk" if not retrieved or score < self.threshold else "answer"), score


def is_idk(answer: str, language: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.strip().lower())
    return normalized == re.sub(r"\s+", " ", IDK[language].strip().lower())


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ask the offline bank FAQ assistant.")
    parser.add_argument("question", help="Question in English, Bangla, or Banglish")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--banglish-rewrite", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    assistant = BankAssistant(
        model=args.model, threshold=args.threshold, banglish_rewrite=args.banglish_rewrite
    )
    print(assistant.ask(args.question)["answer"])

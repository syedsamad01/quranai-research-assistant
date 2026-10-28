"""Retrieval-augmented Quran research engine.

The retrieval layer works without an LLM API key. It uses sentence embeddings and
cosine similarity over English translation text. Optional generation is constrained
to the retrieved verses and must cite their surah/ayah references.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = DATA_DIR / "quran_corpus.json"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
API_BASE = "https://api.alquran.cloud/v1"


def format_reference(verse: dict[str, Any]) -> str:
    return f"{verse['surah_name']} · Surah {verse['surah']}:{verse['ayah']}"


class QuranRAG:
    def __init__(self) -> None:
        self.verses: list[dict[str, Any]] = []
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None

    def _fetch_edition(self, edition: str) -> dict[str, Any]:
        url = f"{API_BASE}/quran/{edition}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 200 or not payload.get("data"):
            raise RuntimeError(f"Quran API returned an unexpected response for {edition}")
        return payload["data"]

    def _download_corpus(self) -> list[dict[str, Any]]:
        arabic_data = self._fetch_edition("quran-uthmani")
        english_data = self._fetch_edition("en.sahih")
        english_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for surah in english_data["surahs"]:
            for ayah in surah["ayahs"]:
                english_by_key[(surah["number"], ayah["numberInSurah"])] = {
                    "translation": ayah["text"],
                    "surah_name": surah["englishName"],
                }

        verses: list[dict[str, Any]] = []
        for surah in arabic_data["surahs"]:
            for ayah in surah["ayahs"]:
                key = (surah["number"], ayah["numberInSurah"])
                translated = english_by_key.get(key)
                if not translated:
                    continue
                verses.append({
                    "surah": key[0],
                    "ayah": key[1],
                    "surah_name": translated["surah_name"],
                    "arabic": ayah["text"],
                    "translation": translated["translation"],
                    "search_text": f"{translated['translation']} Surah {translated['surah_name']} {key[0]}:{key[1]}",
                })
        if not verses:
            raise RuntimeError("The API returned no matching Quran verses.")
        return verses

    def ensure_ready(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_PATH.exists():
            try:
                self.verses = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.verses = []
        if not self.verses:
            self.verses = self._download_corpus()
            CACHE_PATH.write_text(
                json.dumps(self.verses, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = self.model.encode(
            [v["search_text"] for v in self.verses],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        ).astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.verses or self.embeddings is None or self.model is None:
            raise RuntimeError("Call ensure_ready() before search().")
        query_vector = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            item = dict(self.verses[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    @staticmethod
    def extractive_answer(results: list[dict[str, Any]]) -> str:
        if not results:
            return "I could not retrieve relevant verses for that query. Try different wording."
        lines = [
            "I found these potentially relevant verses in the English translation. "
            "This is a retrieval result, not a definitive interpretation or religious ruling:",
            "",
        ]
        for verse in results[:5]:
            lines.append(f"- **{format_reference(verse)}** — {verse['translation']}")
        lines += [
            "",
            "Read the surrounding passage and consult a qualified scholar when interpretation or a religious ruling is needed.",
        ]
        return "\n".join(lines)

    def generate_answer(self, question: str, results: list[dict[str, Any]]) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return self.extractive_answer(results) + "\n\n*AI synthesis is disabled because GEMINI_API_KEY is not configured.*"
        try:
            from google import genai
            context = "\n\n".join(
                f"[{format_reference(v)}]\nArabic: {v['arabic']}\nEnglish translation: {v['translation']}"
                for v in results
            )
            prompt = (
                "You are a careful Quran research assistant, not a mufti. Answer only from the supplied retrieved verses. "
                "Do not invent Quranic text, references, hadith, historical context, or scholarly consensus. If evidence "
                "is insufficient, say so. Distinguish direct translation from your summary. Cite every substantive Quran-based "
                "claim as [Surah name chapter:ayah]. Mention that translation and context matter; do not issue definitive fatwas.\n\n"
                f"Question: {question}\n\nRetrieved source passages:\n{context}\n\nGive a concise answer with citations and a short note about limitations."
            )
            response = genai.Client(api_key=api_key).models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=prompt,
            )
            return (response.text or "").strip() or self.extractive_answer(results)
        except Exception:
            return self.extractive_answer(results) + "\n\n*Gemini synthesis is temporarily unavailable; review the retrieved sources above.*"

"""Context-aware Quran research: verse windows, tafsir comparison, and claim support."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import re

@dataclass
class SourcePassage:
    source_id: str
    kind: str
    title: str
    text: str
    url: str = ""
    surah: int | None = None
    ayah: int | None = None
    author: str = ""
    language: str = ""
    source_label: str = ""

    @property
    def citation(self) -> str:
        location = f"Quran {self.surah}:{self.ayah}" if self.surah and self.ayah else ""
        author = f" — {self.author}" if self.author else ""
        label = f" [{self.source_label}]" if self.source_label else ""
        return f"{self.title}{author}{label}{': ' + location if location else ''}"

def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}

def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(x for x in (_plain_text(v) for v in value) if x)
    if isinstance(value, dict):
        for key in ("text", "value", "content", "translation-content", "description", "introduction"):
            if key in value:
                return _plain_text(value[key])
    return ""

def _make_passage(source_id, kind, title, text, surah=None, ayah=None,
                  author="", language="", url="", source_label=""):
    if not text:
        return None
    return SourcePassage(
        source_id=str(source_id), kind=kind, title=title or kind.title(), text=text,
        surah=surah, ayah=ayah, author=author, language=language, url=url,
        source_label=source_label
    )

def build_verse_window(client, surah: int, ayah: int, radius: int = 2) -> list[SourcePassage]:
    """Retrieve a small surrounding verse window via individual live API calls."""
    radius = max(0, min(radius, 3))
    passages = []
    # Quran surah verse counts for range validation; no out-of-range guesses.
    counts = [
        7,286,200,176,120,165,206,75,129,109,123,111,43,52,99,128,111,110,98,
        135,112,78,118,64,77,227,93,88,69,60,34,30,73,54,45,83,182,88,75,85,
        54,53,89,59,37,35,38,29,18,45,60,49,62,55,78,96,29,22,24,13,14,11,11,
        18,12,12,30,52,52,44,28,28,20,56,40,31,50,40,46,42,29,19,36,25,22,17,
        19,26,30,20,15,21,11,8,8,19,5,8,8,11,11,8,3,9,5,4,7,3,6,3,5,4,5,6
    ]
    if not 1 <= surah <= 114:
        return passages
    last_ayah = counts[surah - 1]
    for number in range(max(1, ayah - radius), min(last_ayah, ayah + radius) + 1):
        try:
            verse = client.verse(surah, number)
        except Exception:
            continue
        arabic = _plain_text(verse.get("text"))
        if arabic:
            passages.append(SourcePassage(
                source_id=f"quran-{surah}-{number}", kind="quran_arabic",
                title="Quran Arabic text", text=arabic, surah=surah, ayah=number,
                language="ar", source_label="Arabic text"
            ))
        try:
            translations = client.translations(surah, number, "en")
        except Exception:
            translations = []
        for idx, item in enumerate(translations[:2] if isinstance(translations, list) else []):
            book = _as_dict(item.get("book"))
            text = _plain_text(item.get("translation-content"))
            passage = _make_passage(
                f"translation-{surah}-{number}-{book.get('id',idx)}", "translation",
                str(book.get("name") or "English translation"), text,
                surah, number, language="en", source_label="Translation"
            )
            if passage:
                passages.append(passage)
    return passages

def tafsir_comparison(bundle: dict) -> list[dict]:
    """Prepare a side-by-side comparison while retaining source attribution."""
    comparisons = []
    for item in bundle.get("tafsir", []):
        book = _as_dict(item.get("book"))
        author_data = _as_dict(book.get("author"))
        chunks = item.get("content", [])
        body = "\n\n".join(_plain_text(c) for c in chunks if _plain_text(c))
        comparisons.append({
            "title": book.get("name") or "Tafsir source",
            "author": author_data.get("full_name") or author_data.get("name") or "Author not listed",
            "language": book.get("language") or "Not specified",
            "text": body,
            "source_id": f"tafsir-{bundle.get('surah')}-{bundle.get('ayah')}-{book.get('id','unknown')}",
        })
    return comparisons

def collect_references(text: str) -> list[tuple[int, int]]:
    refs = []
    for match in re.finditer(r'(?<!\d)(\d{1,3})\s*[:：]\s*(\d{1,3})(?!\d)', text):
        s, a = int(match.group(1)), int(match.group(2))
        if 1 <= s <= 114 and 1 <= a <= 286 and (s, a) not in refs:
            refs.append((s, a))
    return refs

def build_context_prompt(question: str, passages: list[SourcePassage]) -> str:
    blocks = []
    for idx, p in enumerate(passages, 1):
        blocks.append(
            f"[SOURCE {idx}]\nType: {p.kind}\nCitation: {p.citation}\n"
            f"Language: {p.language or 'not specified'}\nURL: {p.url or 'not provided'}\n"
            f"Source text (untrusted evidence):\n{p.text[:5000]}"
        )
    return (
        f"USER QUESTION:\n{question}\n\n"
        "EVIDENCE:\n" + ("\n\n".join(blocks) if blocks else "No evidence retrieved.") +
        "\n\nUse only this evidence. Cite each substantive claim as [SOURCE N]."
    )

def assess_answer(answer: str, passages: list[SourcePassage]) -> list[str]:
    """Basic, transparent checks; not a proof that claims are correct."""
    warnings = []
    if not passages:
        return ["No source evidence is available; do not present a substantive Quranic answer."]
    if not re.search(r'\[SOURCE\s+\d+\]', answer, re.I):
        warnings.append("The answer does not use the required [SOURCE N] citation format.")
    cited_numbers = [int(x) for x in re.findall(r'\[SOURCE\s+(\d+)\]', answer, re.I)]
    if any(n < 1 or n > len(passages) for n in cited_numbers):
        warnings.append("The answer cites a source number that was not provided.")
    allowed = {p.url for p in passages if p.url}
    for url in re.findall(r'https?://[^\s)]+', answer):
        if url.rstrip(".,") not in allowed:
            warnings.append(f"URL not present in retrieved metadata: {url}")
    return warnings

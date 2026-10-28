"""Small, attribution-preserving client for the live Quranpedia API."""
from __future__ import annotations
from typing import Any
from urllib.parse import quote
import requests

BASE_URL = "https://api.quranpedia.net/v1"

class QuranpediaError(RuntimeError):
    pass

class QuranpediaClient:
    def __init__(self, timeout: int = 25):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "QuranAI-Research-Assistant/0.2 (educational research app)",
            "Accept": "application/json",
        })
        self.timeout = timeout

    def get(self, path: str) -> Any:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise QuranpediaError(f"Could not retrieve live source data ({exc}).") from exc
        except ValueError as exc:
            raise QuranpediaError("The source returned an unexpected non-JSON response.") from exc

    def verse(self, surah: int, ayah: int, mushaf_id: int = 1) -> dict:
        return self.get(f"mushafs/{mushaf_id}/{surah}/{ayah}")

    def translations(self, surah: int, ayah: int, language: str = "en") -> list[dict]:
        result = self.get(f"translations/{surah}/{ayah}/{quote(language)}")
        return result if isinstance(result, list) else []

    def available_tafsir_books(self, surah: int) -> list[dict]:
        result = self.get(f"surah/tafsirs/{surah}")
        return result if isinstance(result, list) else []

    def tafsir(self, surah: int, ayah: int, book_id: int) -> dict:
        return self.get(f"ayah/{surah}/{ayah}/book/{book_id}")

    def surah_information(self, surah: int) -> dict:
        return self.get(f"surah/information/{surah}")

    def ayah_options(self, surah: int, ayah: int) -> dict:
        return self.get(f"ayah/{surah}/{ayah}/options")

    def search(self, query: str, kind: str | None = None) -> dict:
        path = f"search/{quote(query, safe='')}"
        if kind and kind != "all":
            path += f"/{quote(kind, safe='')}"
        return self.get(path)

    def topics(self) -> list[dict]:
        result = self.get("topics")
        return result if isinstance(result, list) else []

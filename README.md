# QuranAI Research Assistant

An evidence-first Quran research app built with Python, Streamlit, Sentence Transformers, and optional LLM synthesis. It retrieves Quran verses by semantic similarity and displays Arabic text, an English translation, and surah/ayah references.

> **Scope note:** This is a research and discovery tool, not a religious authority. Retrieval relevance is not proof of interpretation. Always inspect context and consult qualified scholars for religious rulings.

## Features

- Semantic retrieval over Quran English translation text using Sentence Transformers.
- Verse-level references and paired Arabic Uthmani script.
- Optional RAG-style answer synthesis through the Gemini API.
- No API key required for semantic search and extractive results.
- Local JSON cache after the first Quran text download.
- Polished Streamlit interface with sample prompts and conversation history.
- Fallback to source passages if the LLM API is unavailable.

## Architecture

1. **Corpus ingestion:** downloads Arabic Uthmani text and the `en.sahih` translation from AlQuran Cloud API.
2. **Alignment:** pairs Arabic and English verses by surah number and ayah number.
3. **Embedding:** encodes English translation passages using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieval:** normalized embeddings and cosine similarity return top-k verses.
5. **Generation (optional):** the LLM receives only retrieved passages and is instructed to cite references and avoid unsupported claims.
6. **Interface:** Streamlit shows the answer and source verses.

## Requirements

- Python 3.11 or 3.12 recommended.
- Internet access on first launch to download the Quran text and embedding model.
- Optional Gemini API key for generated synthesis. API usage may incur charges.
- The embedding model is downloaded locally on first run; initial startup can take several minutes.

## Run locally (Windows PowerShell)

```powershell
# 1. Open the project folder in VS Code.
# 2. Create a virtual environment.
py -3.12 -m venv .venv

# 3. Activate it.
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies.
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Optional: create .env from the example and add your own API key.
Copy-Item .env.example .env

# 6. Start the app.
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Python 3.14 note

Some machine-learning dependencies may not yet provide compatible wheels for Python 3.14. Use Python 3.12 for the most predictable setup. Keep your existing Python installation; install Python 3.12 alongside it and use `py -3.12` commands.

## Enable answer generation

1. Copy `.env.example` to `.env`.
2. Add your own API key after `GEMINI_API_KEY=`.
3. Restart Streamlit.

Never commit `.env` or expose API keys in screenshots or GitHub. Without a key, the app still performs semantic retrieval and displays source passages.

## Corpus and attribution

The app fetches Quran text from the AlQuran Cloud API:
- API: https://alquran.cloud/api
- Arabic edition requested: `quran-uthmani`
- English translation edition requested: `en.sahih`

Check the provider's current terms, edition metadata, and attribution requirements before redistributing corpus data or deploying commercially. The downloaded corpus cache is excluded from Git. Translation wording and verse numbering can differ across editions and counting conventions.

## Project structure

```text
quranai-research-assistant/
├── app.py
├── quran_rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/                 # local cache; not committed
```

## Known limitations

- The current retrieval corpus uses English translation text, not Arabic morphology or tafsir.
- Semantic similarity is a ranking signal, not a measure of religious correctness.
- The generator can still make mistakes; verify all references and claims.
- This version does not provide hadith retrieval, a tafsir corpus, fatwa generation, or tajweed evaluation.
- For production use, add evaluation tests, source edition metadata, rate limits, observability, and expert review.

## Suggested next milestones

- Add a hybrid lexical + vector retrieval mode.
- Add surah filters and translation selection.
- Build an evaluation set of questions with scholar-reviewed relevant references.
- Add a transparent retrieval panel with similarity scores.
- Deploy on Streamlit Community Cloud after verifying dataset/API and model terms.

## License

Choose a project license after reviewing the licenses and terms for every dependency, model, text edition, and translation. Do not assume all Quran translations have identical reuse terms.

## Open-source local voice mode

The **Voice chat** tab uses **faster-whisper** for local speech-to-text and **Piper** for local text-to-speech. These do not require a Gemini API key. The first Whisper run downloads the selected model, and performance depends on your CPU/RAM.

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Download a Piper voice from the [Piper voices collection](https://huggingface.co/rhasspy/piper-voices), including both the `.onnx` model and its matching `.onnx.json` config. Check the voice/model license before redistributing it.
3. Set the full `.onnx` path in `.env`, for example `PIPER_VOICE_MODEL=C:\\models\\en_US-lessac-medium.onnx`. Keep the matching `.onnx.json` beside it.
4. Ensure the Piper executable is available in the virtual environment; `PIPER_EXECUTABLE=piper` is the default.
5. Start with `python -m streamlit run app.py`, allow microphone access, and choose **Transcribe locally**.

`WHISPER_MODEL=base` is a modest starting point for CPU use; `tiny` is faster but less accurate, while `small` can improve recognition at higher memory/runtime cost. These local models may need internet for the first download, but processing happens on your computer afterwards. Quran/tafsir research still requires internet access. Generated speech is for explanations, not Quran recitation. Use verified reciter audio for recitation.

Gemini remains optional for text synthesis if configured, but is not used for microphone transcription or speech playback in this local voice workflow.

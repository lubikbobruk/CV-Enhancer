# CV_Enhancer

## 🚀 Project Roadmap

Phase 0 — Setup: repo, env, deps, hello-world Streamlit + Gemini
Phase 1 — Parsing: PDF/DOCX → text → chunks
Phase 2 — Classical retrieval: TF-IDF retriever
Phase 3 — Dense retrieval + fusion: sentence-transformers + RRF
Phase 4 — LLM generation: prompts, schemas, structured Gemini output
Phase 5 — Streamlit UI: full user flow + PDF download
Phase 6 — Evaluation: precision@k harness on labeled pairs
Phase 7 — Polish + deploy: README, demo video, Streamlit Cloud
Phase 8 — Report: write the academic deliverable

## Setup

Requires [conda](https://docs.conda.io/projects/miniconda/) and a Gemini API key from https://aistudio.google.com/apikey.

1. Create the conda env (installs Python 3.11 and all pip dependencies from `requirements.txt`):
   ```
   conda env create -f environment.yml
   ```
2. Activate it:
   ```
   conda activate cv-enhancer
   ```
3. Configure your API key:
   ```
   cp .env.example .env
   ```
   Then open `.env` and set `GEMINI_API_KEY=your_key_here`.
4. Run the app:
   ```
   streamlit run app.py
   ```

To update an existing env after `requirements.txt` changes:
```
conda env update -f environment.yml --prune
```
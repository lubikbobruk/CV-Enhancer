# 📄 CV Enhancer

A Python application that rewrites a candidate's CV to better match a target job advertisement. It combines classical and dense information retrieval with LLM support to produce an enhanced CV tailored to the role, along with structured feedback explaining what was changed and why.

## 🚀 Project Overview

Upload a CV (PDF/DOCX), paste a job ad, click **Enhance**. The app parses and chunks the document, retrieves the most and least relevant sections using **TF-IDF**, **dense embeddings**, and **RRF hybrid fusion**, then feeds everything into **Gemini 2.5 Flash** to rewrite the CV for the target role. The enhanced CV is downloadable as a styled PDF. A feedback panel shows strengths, gaps, and what was changed.

Current status: **In development**: Phase 4

## 🔑 API Setup

Requires a **Google Gemini** API key. Not included in the repository.

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
GEMINI_API_KEY=your_key_here
```

Get your key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## 🛠️ Installation

### Prerequisites

- [Anaconda / Miniconda](https://docs.conda.io/projects/miniconda/)
- Git

### Setup

```bash
conda env create -f environment.yml
conda activate cv-enhancer
```

### Run

```bash
streamlit run app.py
```

### Dev utilities

```bash
python -m pytest tests/    # unit tests
flake8 .                   # pep8 audit
```

## 🗂️ Project Structure

```
cv-enhancer/
├── app.py                        # Streamlit entry point
├── .env.example                  # API key template
├── environment.yml               # Conda env
├── requirements.txt              # Python dependencies
├── src/
│   ├── config.py                 # Env loading, constants
│   ├── parsing/                  # Parser and Chunker
│   │   ├── __init__.py           
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   └── generation/
│       ├── gemini_client.py
│       ├── prompts.py
│       └── schemas.py           
├── evaluation/
│   ├── dataset.py
│   ├── metrics.py                # P@k, MRR
│   └── run_eval.py
├── tests/                        # Unit testing
└── data/
```

## 🚀 Roadmap

- **Phase 0 — Setup:** Repo, environment, Streamlit + Gemini hello-world. ✅
- **Phase 1 — Parsing:** PDF/DOCX → text → chunks. ✅
- **Phase 2 — Classical Retrieval:** TF-IDF retriever wrapping sklearn. ✅
- **Phase 3 — Dense Retrieval + Fusion:** sentence-transformers + RRF. ✅
- **Phase 4 — LLM Generation:** Structured Gemini output (enhanced CV + feedback).
- **Phase 5 — Streamlit UI:** Full flow — upload, enhance, download PDF.
- **Phase 6 — Evaluation:** Precision@k / MRR on labeled pairs.
- **Phase 7 — Polish + Deploy:** Demo video, Streamlit Cloud.
- **Phase 8 — Report:** Academic deliverable.

### Future work

- Skill taxonomy mapping (ESCO)
- ATS compatibility scoring
- Cover letter generation
- Job ad aggregation from boards
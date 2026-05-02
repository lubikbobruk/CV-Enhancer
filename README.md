# 📄 CV Enhancer

![CV Enhancer demo](preview.gif)

A Python application that rewrites a candidate's CV to better match a target job advertisement. It combines classical and dense information retrieval with LLM support to produce an enhanced CV tailored to the role, along with structured feedback explaining what was changed and why.

## 🚀 Project Overview

Upload a CV (PDF/DOCX), paste a job ad, click **Enhance**. The app parses and chunks the document, retrieves the most and least relevant sections using **TF-IDF**, **dense embeddings**, and **RRF hybrid fusion**, then feeds everything into **Gemini 2.5 Flash** to rewrite the CV for the target role. The enhanced CV is downloadable as a styled PDF. A feedback panel shows strengths, gaps, and what was changed.

Current status: **In development**: Phase 7

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
python -m tests.smoke_phase4        # smoke test for llm enhancer on fixture
pytest                              # unit tests
flake8 .                            # pep8 audit
```

## 🗂️ Project Structure

```
cv-enhancer/
├── app.py                        # Streamlit entry point
├── .env.example                  # API key template
├── src/
│   ├── parsing/                  # PDF/DOCX parsing and chunking
│   ├── retrieval/                # TF-IDF, dense, RRF fusion
│   ├── generation/               # Gemini client, prompts, schemas
│   └── output/                   # PDF writer
├── evaluation/                   # Precision@k and MRR
├── tests/                        # Unit tests and fixtures
└── data/                         # Eval datasets
```

## 🚀 Roadmap

- **Phase 0 — Setup:** Repo, environment, Streamlit + Gemini hello-world. ✅
- **Phase 1 — Parsing:** PDF/DOCX → text → chunks. ✅
- **Phase 2 — Classical Retrieval:** TF-IDF retriever wrapping sklearn. ✅
- **Phase 3 — Dense Retrieval + Fusion:** sentence-transformers + RRF. ✅
- **Phase 4 — LLM Generation:** Structured Gemini output (enhanced CV + feedback). ✅
- **Phase 5 — Streamlit UI:** Full flow — upload, enhance, download PDF. ✅
- **Phase 6 — Polish + Deploy:** Demo video, Streamlit Cloud. ✅
- **Phase 7 — Report:** Academic deliverable.
- **Phase 8 — Evaluation:** Precision@k / MRR on labeled pairs.

### Future work

- Skill taxonomy mapping (ESCO)
- ATS compatibility scoring
- Cover letter generation
- Job ad aggregation from boards
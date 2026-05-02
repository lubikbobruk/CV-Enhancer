# 📄 CV Enhancer

Rewrite a candidate's CV to better match a target job advertisement. Combines classical and dense information retrieval with a constrained LLM rewrite, end-to-end from a PDF/DOCX upload to a styled PDF download.

**Live demo:** [cv-enhancer.streamlit.app](https://cv-enhancer.streamlit.app/)

![CV Enhancer demo](preview.gif)

## 🚀 Project Overview

Upload a CV (PDF/DOCX), paste a job ad, click **Enhance**:

1. **Parse + chunk** the document.
2. **Filter the job ad** with a nearest-centroid classifier (MiniLM anchors) to drop benefits/contact parts of the job ad.
3. **Rank chunks** against the filtered ad using three retrievers in parallel:
   - **TF-IDF** — hand-implemented vector-space model.
   - **Dense** — `sentence-transformers/all-MiniLM-L6-v2` cosine via dot product.
   - **RRF** — Reciprocal Rank Fusion (`k_rrf=60`) over both rankings.
4. **Rewrite** user-selected chunks with **Gemini 2.5 Flash** under a strict system prompt and a Pydantic-typed structured output.
5. **Filter regressions** — re-embed each rewrite and drop any that didn't improve dense cosine vs. the ad.
6. **Render** the enhanced CV to a styled PDF formatting, with option to apply specific rewritten chunks and a photo (if initialy was one).

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
- Python 3.11

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
├── app.py                          # Local streamlit entry point
├── .env.example                    # API key template
├── CV_Enhancer_Report_BI-VWM.pdf   # Academic report
├── src/
│   ├── parsing/                    # PDF/DOCX parsing and chunking
│   ├── retrieval/                  # TF-IDF, dense, RRF fusion
│   ├── generation/                 # Gemini client, prompts, schemas
│   ├── ui/                         # Streamlit frontend
│   └── output/                     # PDF writer
├── visualization/                  # Notebook with visualized results
└── tests/                          # Unit tests and fixtures
```

## 🚀 Roadmap

- **Phase 0 — Setup:** Repo, environment, Streamlit + Gemini hello-world. ✅
- **Phase 1 — Parsing:** PDF/DOCX → text → chunks. ✅
- **Phase 2 — Classical Retrieval:** TF-IDF retriever wrapping sklearn. ✅
- **Phase 3 — Dense Retrieval + Fusion:** sentence-transformers + RRF. ✅
- **Phase 4 — LLM Generation:** Structured Gemini output (enhanced CV + feedback). ✅
- **Phase 5 — Streamlit UI:** Full flow — upload, enhance, download PDF. ✅
- **Phase 6 — Polish + Deploy:** Demo video, Streamlit Cloud. ✅
- **Phase 7 — Report:** Academic deliverable. ✅

### Future work
- Evaluation with Precision@k / MRR on labeled pairs
- Parsing enhancement

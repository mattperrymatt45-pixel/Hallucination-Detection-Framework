# RAG Hallucination Detection & Self-Consistency Evaluation Framework

A production-grade Python framework that detects potential hallucinations in Large Language Model (LLM) answers by grounding them against retrieved evidence and measuring cross-sample self-consistency.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────────────┐
│  Gemini LLM │────▶│  Primary Answer           │
│  (llm.py)   │     └──────────────┬───────────┘
│             │────▶│  N Alternative Answers    │
└─────────────┘     └──────────────┬───────────┘
                                   │
┌─────────────────────────┐        │
│  FAISS Retriever        │        │
│  (retriever.py)         │────────┤
│  · Load .txt docs       │        │
│  · Chunk (500 chars)    │        ▼
│  · Embed (MiniLM-L6-v2)│   ┌──────────────────────────────┐
│  · Top-3 passages       │   │  Hallucination Scorer        │
└─────────────────────────┘   │  (scorer.py)                 │
                               │  · Cosine similarity         │
                               │  · Evidence support score    │
                               │  · Hallucination score       │
                               │  · Self-consistency score    │
                               │  · Final confidence (0–100)  │
                               │  · Risk classification       │
                               └──────────────┬───────────────┘
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │  Streamlit UI     │
                                   │  (app.py)         │
                                   └──────────────────┘
```

### Scoring Pipeline

| Component | Formula | Weight |
|---|---|---|
| Cosine Similarity | `cosine(answer_vec, centroid(evidence_vecs))` | — |
| Evidence Support | `mean(cosine(answer_vec, each_evidence_vec))` | — |
| Hallucination Score | `(cosine_sim × 0.5 + support × 0.5) × 100` | 60 % |
| Self-Consistency | `mean_pairwise_cosine(alt_answers) × 100` | 40 % |
| **Final Confidence** | `hallucination × 0.6 + consistency × 0.4` | — |

### Risk Classification

| Score | Risk Level |
|---|---|
| 0 – 40 | 🔴 High Risk |
| 41 – 70 | 🟡 Medium Risk |
| 71 – 100 | 🟢 Low Risk |

---

## Installation

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Steps

```bash
# 1. Clone / download the project
git clone <repo-url>
cd project

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY

# 5. Add your knowledge documents
# Place one or more .txt files inside the docs/ folder.
# A sample knowledge_base.txt is included to get started.
```

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
project/
├── app.py           # Streamlit UI
├── llm.py           # Gemini API integration
├── retriever.py     # FAISS document retriever
├── scorer.py        # Hallucination & consistency scorer
├── utils.py         # Shared helpers
├── docs/
│   └── knowledge_base.txt   # Sample knowledge documents
├── requirements.txt
├── .env.example
└── README.md
```

---

## Adding Custom Knowledge

1. Create `.txt` files with your domain content.
2. Place them in the `docs/` directory.
3. Restart the app — the index rebuilds automatically on startup.

Multiple files are supported and will be merged into a single FAISS index.

---

## Future Improvements

- **Overlap chunking** — use sliding-window chunks with configurable overlap to reduce boundary information loss.
- **Re-ranker** — add a cross-encoder re-ranking stage after FAISS retrieval to improve evidence quality.
- **Claim-level scoring** — decompose the answer into atomic claims and score each claim independently.
- **Persistent index** — serialise the FAISS index to disk to avoid re-embedding on every restart.
- **Multi-modal support** — extend retrieval to PDF and web sources.
- **Streaming answers** — stream Gemini responses token-by-token to the Streamlit UI.
- **Feedback loop** — allow users to flag incorrect answers to refine the knowledge base.
- **Multiple LLM backends** — abstract `llm.py` behind a provider interface to support OpenAI, Anthropic, etc.
- **Confidence calibration** — empirically calibrate score thresholds against labelled hallucination datasets.

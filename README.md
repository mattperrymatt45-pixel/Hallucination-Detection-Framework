# 🔍 Hallucination Detection & Self-Consistency Framework

A Retrieval-Augmented evaluation system for detecting hallucinations in LLM outputs using semantic grounding and cross-sample consistency analysis.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![OpenRouter](https://img.shields.io/badge/OpenRouter-API%20Compatible-purple)

---

## 🚀 Live Demo

Try the deployed application here:
**[https://hallucination-detection-framework-4vdvxamagy4cpmdewl6aea.streamlit.app/](https://hallucination-detection-framework-4vdvxamagy4cpmdewl6aea.streamlit.app/)**

---

## 📌 Overview

Large Language Models generate fluent but sometimes factually incorrect outputs — known as **hallucinations**. This framework provides a lightweight, modular evaluation layer that:

- **Grounds answers** using retrieval-augmented evidence (FAISS)
- **Measures semantic alignment** with source documents
- **Evaluates self-consistency** across multiple LLM samples
- **Produces an interpretable hallucination risk score** (0–100)

Designed for:
- AI safety evaluation
- RAG system benchmarking
- Enterprise LLM validation pipelines
- Research on hallucination mitigation

> **Key Idea:** Instead of trusting a single model output, the system asks: *does the answer align with external knowledge and remain stable across multiple generations?*

---

## 🏗️ Architecture

```
User Query (Q)
      │
      ▼
LLM Generation Layer
  ├── Primary Answer
  └── N Alternative Answers
      │
      ▼
Retrieval-Augmented Layer
  ├── FAISS Vector Store
  ├── Semantic Chunking
  └── Top-k Evidence Retrieval
      │
      ▼
Evaluation & Scoring Engine
  ├── Evidence Similarity Score
  ├── Self-Consistency Score
  └── Hallucination Risk Model
      │
      ▼
Streamlit Dashboard
  ├── Risk Visualization
  └── Confidence Score
```

---

## 📐 Scoring Framework

### 1. Evidence Alignment
Measures how well the answer matches retrieved knowledge using cosine similarity and aggregated evidence support.

### 2. Hallucination Score

$$H = 100 \cdot (0.5 \cdot S_{\cos} + 0.5 \cdot S_{\sup})$$

### 3. Self-Consistency Score

$$C = 100 \cdot \frac{1}{n(n-1)} \sum_{i \neq j} \cos(v(A_i), v(A_j))$$

### 4. Final Confidence Score

$$F = 0.6H + 0.4C$$

### Risk Levels

| Score | Interpretation |
|-------|----------------|
| 0 – 40 | 🔴 High hallucination risk |
| 41 – 70 | 🟡 Medium risk / partially grounded |
| 71 – 100 | 🟢 Low risk / well supported |

---

## 🗂️ Project Structure

```
Hallucination-Detection-Framework/
│
├── app.py              # Streamlit UI layer
├── llm.py              # OpenRouter LLM integration
├── retriever.py        # FAISS retrieval engine
├── scorer.py           # Hallucination scoring logic
├── utils.py            # Shared utilities
│
├── docs/
│   └── knowledge_base.txt
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd Hallucination-Detection-Framework
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## 🔌 OpenRouter Configuration

The project uses [OpenRouter's](https://openrouter.ai) OpenAI-compatible API endpoint:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

**Recommended models:**

| Model | Use Case |
|-------|----------|
| `mistralai/mistral-7b-instruct` | Fast, cost-efficient |
| `meta-llama/llama-3.1-8b-instruct` | Open-source baseline |
| `openai/gpt-4o-mini` | Higher accuracy |

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## ✨ Core Features

### Retrieval-Augmented Grounding
- FAISS-based semantic search
- SentenceTransformer embeddings
- Chunked document indexing

### Multi-Sample LLM Evaluation
- Primary + multiple alternative responses
- Stability analysis across generations

### Interpretability Layer
- Evidence alignment scoring
- Self-consistency estimation
- Normalized risk classification

### Interactive Dashboard
- Streamlit-based UI
- Real-time hallucination scoring
- Confidence visualization

---

## 🧪 Use Cases

- RAG system evaluation
- LLM reliability benchmarking
- AI safety auditing
- Enterprise knowledge assistants
- Academic research on hallucination detection

---

## 🗺️ Roadmap

- [ ] Sliding window chunking with overlap
- [ ] Cross-encoder reranking for retrieval quality
- [ ] Claim-level hallucination detection
- [ ] Persistent FAISS index storage
- [ ] Multi-modal ingestion (PDF, web, images)
- [ ] Streaming response UI
- [ ] Feedback loop for correction learning
- [ ] Multi-provider model abstraction layer
- [ ] Calibration on labeled hallucination datasets

---

## 📄 License

This project is open-source. See [LICENSE](./LICENSE) for details.

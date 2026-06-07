"""
app.py — Streamlit front-end for the Hallucination Detection Framework.
"""

import logging

import streamlit as st

from llm import generate_answer, generate_multiple_answers
from retriever import DocumentRetriever
from scorer import HallucinationScorer, ScoreResult
from utils import configure_logging, risk_colour, score_gauge_label, truncate

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hallucination Detection Framework",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading document index…")
def load_retriever() -> DocumentRetriever:
    retriever = DocumentRetriever(docs_dir="docs")
    n = retriever.build_index()
    logger.info("Retriever ready — %d chunks indexed.", n)
    return retriever


@st.cache_resource(show_spinner="Loading scoring model…")
def load_scorer() -> HallucinationScorer:
    return HallucinationScorer()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_score_card(label: str, value: float, suffix: str = "") -> None:
    st.metric(label=label, value=f"{value:.1f}{suffix}")


def _render_risk_badge(risk_level: str) -> None:
    colour = risk_colour(risk_level)
    st.markdown(
        f"<span style='background-color:{colour};color:white;"
        f"padding:6px 16px;border-radius:20px;"
        f"font-weight:700;font-size:1rem;'>{risk_level}</span>",
        unsafe_allow_html=True,
    )


def _render_evidence(passages: list[str]) -> None:
    if not passages:
        st.info("No supporting passages retrieved — docs/ folder may be empty.")
        return
    for i, passage in enumerate(passages, start=1):
        with st.expander(f"📄 Evidence Passage {i}"):
            st.write(truncate(passage, max_chars=600))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🔍 RAG Hallucination Detection & Self-Consistency Framework")
    st.caption(
        "Detects potential hallucinations in LLM answers by comparing them "
        "against retrieved evidence and measuring answer self-consistency."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        n_answers = st.slider(
            "Independent answers for consistency check",
            min_value=2,
            max_value=5,
            value=3,
        )
        top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=5, value=3)
        st.markdown("---")
        st.markdown(
            "**Risk bands**\n"
            "- 🔴 0–40 · High Risk\n"
            "- 🟡 41–70 · Medium Risk\n"
            "- 🟢 71–100 · Low Risk"
        )

    # Load resources
    try:
        retriever = load_retriever()
        scorer = load_scorer()
    except FileNotFoundError as exc:
        st.error(f"Initialisation error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error during startup: {exc}")
        st.stop()

    # Input
    st.markdown("### Ask a Question")
    question = st.text_area(
        label="Question",
        placeholder="e.g. What are the main causes of climate change?",
        height=100,
        label_visibility="collapsed",
    )

    run_btn = st.button("🚀 Analyse", type="primary", use_container_width=True)

    if not run_btn:
        return

    if not question.strip():
        st.warning("Please enter a question before clicking Analyse.")
        return

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    with st.status("Running analysis pipeline…", expanded=True) as status:

        # Step 1 — primary answer
        st.write("🤖 Generating primary answer…")
        try:
            primary_answer = generate_answer(question)
        except (EnvironmentError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        # Step 2 — retrieve evidence
        st.write("📚 Retrieving supporting evidence…")
        try:
            evidence = retriever.get_evidence_text(question, top_k=top_k)
        except Exception as exc:
            logger.warning("Evidence retrieval failed: %s", exc)
            evidence = []

        # Step 3 — generate n alternative answers
        st.write(f"🔄 Generating {n_answers} independent answers for consistency check…")
        try:
            alt_answers = generate_multiple_answers(question, n=n_answers)
        except (EnvironmentError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        # Step 4 — score
        st.write("📊 Computing hallucination & consistency scores…")
        result: ScoreResult = scorer.score(
            answer=primary_answer,
            evidence_passages=evidence,
            alternative_answers=alt_answers,
        )

        status.update(label="Analysis complete ✅", state="complete", expanded=False)

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------
    st.markdown("---")
    col_answer, col_scores = st.columns([3, 2], gap="large")

    with col_answer:
        st.markdown("### 💬 Generated Answer")
        st.write(primary_answer)

        st.markdown("### 📜 Retrieved Evidence")
        _render_evidence(evidence)

    with col_scores:
        st.markdown("### 📊 Confidence & Risk")

        # Risk badge
        _render_risk_badge(result.risk_level)
        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge label
        st.info(score_gauge_label(result.final_confidence_score))

        # Metrics
        m1, m2 = st.columns(2)
        with m1:
            _render_score_card("Final Confidence", result.final_confidence_score, " / 100")
            _render_score_card("Self-Consistency", result.self_consistency_score, " / 100")
        with m2:
            _render_score_card("Hallucination Score", result.hallucination_score, " / 100")
            _render_score_card("Evidence Support", result.evidence_support_score * 100, " / 100")

        # Cosine similarity detail
        with st.expander("🔬 Raw Similarity Scores"):
            st.write(f"**Answer ↔ Evidence Cosine Similarity:** `{result.cosine_similarity:.4f}`")
            st.write(f"**Evidence Support Score (mean):** `{result.evidence_support_score:.4f}`")
            st.write(
                f"**Self-Consistency Score:** `{result.self_consistency_score:.2f} / 100`"
            )

        # Alternative answers
        with st.expander(f"🔄 {n_answers} Independent Answers"):
            for i, ans in enumerate(alt_answers, start=1):
                st.markdown(f"**Answer {i}:**")
                st.write(ans)
                if i < len(alt_answers):
                    st.markdown("---")


if __name__ == "__main__":
    main()

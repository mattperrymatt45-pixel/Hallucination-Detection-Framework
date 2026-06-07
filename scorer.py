"""
scorer.py — Scoring engine for hallucination detection and self-consistency evaluation.
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Risk band boundaries (inclusive)
_HIGH_RISK_MAX = 40
_MEDIUM_RISK_MAX = 70

# Weighting for final confidence
_RETRIEVAL_WEIGHT = 0.6
_CONSISTENCY_WEIGHT = 0.4


@dataclass
class ScoreResult:
    cosine_similarity: float          # answer ↔ combined evidence
    evidence_support_score: float     # mean per-passage similarity
    hallucination_score: float        # 0–100; higher = less hallucination
    self_consistency_score: float     # 0–100
    final_confidence_score: float     # 0–100
    risk_level: str                   # "High Risk" | "Medium Risk" | "Low Risk"


class HallucinationScorer:
    """
    Computes hallucination and self-consistency scores using embedding-based
    cosine similarity.
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer(_EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        answer: str,
        evidence_passages: list[str],
        alternative_answers: list[str],
    ) -> ScoreResult:
        """
        Compute a full score bundle for one (answer, evidence, alternatives) triple.

        Args:
            answer: Primary generated answer.
            evidence_passages: Retrieved supporting passages.
            alternative_answers: n independently generated answers for consistency check.

        Returns:
            A populated ScoreResult dataclass.
        """
        cosine_sim = self._answer_evidence_cosine(answer, evidence_passages)
        support_score = self._evidence_support_score(answer, evidence_passages)
        hallucination_score = self._compute_hallucination_score(cosine_sim, support_score)
        consistency_score = self._self_consistency_score(alternative_answers)
        final_confidence = self._final_confidence(hallucination_score, consistency_score)
        risk = self._classify_risk(final_confidence)

        return ScoreResult(
            cosine_similarity=round(cosine_sim, 4),
            evidence_support_score=round(support_score, 4),
            hallucination_score=round(hallucination_score, 2),
            self_consistency_score=round(consistency_score, 2),
            final_confidence_score=round(final_confidence, 2),
            risk_level=risk,
        )

    # ------------------------------------------------------------------
    # Scoring sub-components
    # ------------------------------------------------------------------

    def _answer_evidence_cosine(
        self, answer: str, passages: list[str]
    ) -> float:
        """
        Cosine similarity between the answer embedding and the centroid
        of all evidence embeddings.  Returns 0.0 if no passages provided.
        """
        if not passages:
            return 0.0

        texts = [answer] + passages
        embeddings: np.ndarray = self._embed(texts)

        answer_vec = embeddings[0]
        evidence_vecs = embeddings[1:]
        centroid = evidence_vecs.mean(axis=0)

        return float(self._cosine(answer_vec, centroid))

    def _evidence_support_score(
        self, answer: str, passages: list[str]
    ) -> float:
        """
        Mean cosine similarity between the answer and each individual
        evidence passage.  Returns 0.0 if no passages provided.
        """
        if not passages:
            return 0.0

        texts = [answer] + passages
        embeddings: np.ndarray = self._embed(texts)

        answer_vec = embeddings[0]
        scores = [
            float(self._cosine(answer_vec, embeddings[i + 1]))
            for i in range(len(passages))
        ]
        return float(np.mean(scores))

    def _compute_hallucination_score(
        self, cosine_sim: float, support_score: float
    ) -> float:
        """
        Combine cosine similarity and evidence support into a 0–100 score.
        Higher values indicate better grounding (less hallucination).
        """
        raw = (cosine_sim * 0.5 + support_score * 0.5)
        return float(np.clip(raw * 100, 0.0, 100.0))

    def _self_consistency_score(self, answers: list[str]) -> float:
        """
        Compute self-consistency as the mean pairwise cosine similarity
        across all pairs of independently generated answers (0–100 scale).
        """
        valid = [a for a in answers if a.strip()]
        if len(valid) < 2:
            logger.warning("Not enough valid answers for consistency check.")
            return 50.0

        embeddings: np.ndarray = self._embed(valid)
        pairs = list(combinations(range(len(embeddings)), 2))
        if not pairs:
            return 50.0

        similarities = [
            float(self._cosine(embeddings[i], embeddings[j]))
            for i, j in pairs
        ]
        mean_sim = float(np.mean(similarities))
        return float(np.clip(mean_sim * 100, 0.0, 100.0))

    def _final_confidence(
        self,
        hallucination_score: float,
        consistency_score: float,
    ) -> float:
        """
        Weighted combination of retrieval confidence (hallucination_score)
        and self-consistency confidence.
        """
        retrieval_confidence = hallucination_score
        consistency_confidence = consistency_score
        combined = (
            _RETRIEVAL_WEIGHT * retrieval_confidence
            + _CONSISTENCY_WEIGHT * consistency_confidence
        )
        return float(np.clip(combined, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_risk(score: float) -> str:
        """
        Classify confidence score into a risk band.

        Bands:
            0 – 40  → High Risk
            41 – 70 → Medium Risk
            71 – 100 → Low Risk
        """
        if score <= _HIGH_RISK_MAX:
            return "High Risk"
        if score <= _MEDIUM_RISK_MAX:
            return "Medium Risk"
        return "Low Risk"

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for L2-normalised vectors (= dot product)."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

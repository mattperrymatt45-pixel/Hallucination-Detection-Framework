"""
utils.py — Shared utility functions.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger to stdout with a standard format."""
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def truncate(text: str, max_chars: int = 300) -> str:
    """Return text truncated to max_chars with an ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def risk_colour(risk_level: str) -> str:
    """Map a risk level string to a Streamlit-compatible hex colour."""
    mapping = {
        "High Risk": "#e74c3c",
        "Medium Risk": "#f39c12",
        "Low Risk": "#27ae60",
    }
    return mapping.get(risk_level, "#95a5a6")


def score_gauge_label(score: float) -> str:
    """Return a human-readable confidence label for a 0–100 score."""
    if score <= 40:
        return "Low confidence — answer may be hallucinated"
    if score <= 70:
        return "Moderate confidence — partial evidence support"
    return "High confidence — well-supported answer"

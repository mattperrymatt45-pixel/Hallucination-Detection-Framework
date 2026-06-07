"""
llm.py — Gemini API integration for answer generation.
"""

import os
import logging
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.5-flash"
_GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.7,
    max_output_tokens=512,
)
_VARIATION_CONFIG = genai.GenerationConfig(
    temperature=0.9,
    max_output_tokens=512,
)


def _get_model() -> genai.GenerativeModel:
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=_MODEL_NAME)


def generate_answer(question: str) -> str:
    """
    Generate a single answer for the given question using Gemini.

    Args:
        question: The user's input question.

    Returns:
        A string answer from the model.

    Raises:
        EnvironmentError: If the API key is missing.
        RuntimeError: If the API call fails.
    """
    try:
        model = _get_model()
        prompt = (
            "Answer the following question concisely and accurately.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        response = model.generate_content(
            prompt,
            generation_config=_GENERATION_CONFIG,
        )
        return response.text.strip()
    except EnvironmentError:
        raise
    except Exception as exc:
        logger.error("Gemini generate_answer failed: %s", exc)
        raise RuntimeError(f"Failed to generate answer: {exc}") from exc


def generate_multiple_answers(question: str, n: int = 3) -> list[str]:
    """
    Generate n independent answers for the same question with higher temperature
    to introduce variation, enabling self-consistency evaluation.

    Args:
        question: The user's input question.
        n: Number of independent answers to generate (default 3).

    Returns:
        A list of n answer strings.

    Raises:
        EnvironmentError: If the API key is missing.
        RuntimeError: If one or more API calls fail.
    """
    answers: list[str] = []
    model = _get_model()
    prompt = (
        "Answer the following question concisely and accurately.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    for i in range(n):
        try:
            response = model.generate_content(
                prompt,
                generation_config=_VARIATION_CONFIG,
            )
            answers.append(response.text.strip())
        except Exception as exc:
            logger.warning("Attempt %d/%d failed: %s", i + 1, n, exc)
            answers.append("")

    valid = [a for a in answers if a]
    if not valid:
        raise RuntimeError("All attempts to generate multiple answers failed.")

    # Pad with copies of the first valid answer if some calls failed
    while len(answers) < n:
        answers.append(valid[0])

    return answers[:n]

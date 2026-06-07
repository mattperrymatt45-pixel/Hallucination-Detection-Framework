"""
llm.py — Gemini API integration for answer generation.
"""

import os
import streamlit as st
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
    """
    Load Gemini API key from:
    1. Local .env file
    2. Streamlit Cloud Secrets
    """

    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to .env locally "
                "or Streamlit Secrets when deployed."
            )

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name=_MODEL_NAME
    )


def generate_answer(question: str) -> str:
    """
    Generate a single answer for the given question.
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

        raise RuntimeError(
            f"Failed to generate answer: {exc}"
        ) from exc


def generate_multiple_answers(question: str, n: int = 3) -> list[str]:
    """
    Generate multiple answers using ONE Gemini API call.

    This dramatically reduces quota usage compared
    to making n separate API calls.
    """

    try:
        model = _get_model()

        prompt = f"""
Generate {n} different answers to the following question.

Question:
{question}

Return exactly in this format:

ANSWER 1:
...

ANSWER 2:
...

ANSWER 3:
...
"""

        response = model.generate_content(
            prompt,
            generation_config=_VARIATION_CONFIG,
        )

        text = response.text.strip()

        answers: list[str] = []

        sections = text.split("ANSWER ")

        for section in sections[1:]:
            parts = section.split(":", 1)

            if len(parts) == 2:
                answer = parts[1].strip()

                if answer:
                    answers.append(answer)

        if not answers:
            raise RuntimeError(
                "Gemini returned no parseable answers."
            )

        while len(answers) < n:
            answers.append(answers[0])

        return answers[:n]

    except EnvironmentError:
        raise

    except Exception as exc:
        logger.error(
            "Gemini generate_multiple_answers failed: %s",
            exc,
        )

        raise RuntimeError(
            f"Failed to generate multiple answers: {exc}"
        ) from exc

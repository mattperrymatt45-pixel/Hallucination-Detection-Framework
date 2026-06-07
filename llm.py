import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek/deepseek-chat-v3-0324:free"

def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        api_key = st.secrets["OPENROUTER_API_KEY"]

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def generate_answer(question: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content


def generate_multiple_answers(question: str, n: int = 3):
    client = get_client()

    prompt = f"""
Generate {n} different answers.

Question:
{question}

Format:

ANSWER 1:
...

ANSWER 2:
...

ANSWER 3:
...
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,
    )

    text = response.choices[0].message.content

    answers = []

    for block in text.split("ANSWER ")[1:]:
        parts = block.split(":", 1)

        if len(parts) == 2:
            answers.append(parts[1].strip())

    return answers[:n]
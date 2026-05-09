# gpt_client.py — OpenAI GPT API wrapper (stub)


import os

import openai  # type: ignore


def call_gpt_api(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR] OPENAI_API_KEY not set."
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return str(response.choices[0].message["content"]).strip()
    except Exception as e:
        return f"[ERROR] GPT API call failed: {e}"

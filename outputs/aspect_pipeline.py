"""
Aspect-based sentiment + escalation pipeline.

This is the real pipeline your Streamlit app calls. It sends each review to
Google Gemini with the prompt template below and parses the JSON it
returns. To run this for real, set GEMINI_API_KEY as an environment
variable and install the google-generativeai package (`pip install google-generativeai`).

In this sandbox there was no API key available, so llm_aspect_data.py
contains pre-computed outputs (produced by Claude reasoning directly over
the same 35 reviews) standing in for what this function would return live.
Swap USE_STUB = False once you have a key and it calls the real API.
"""
import json
import os


# Load `.end` file (simple key=value parser) so users can store keys there.
# Values already present in the environment are not overridden.
def _load_end_file(path: str = ".end") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Don't crash import if parsing fails; user can check file manually.
        pass


# Try to load .env first (user renamed) then fall back to .end at import time
_load_end_file(path=".env")
_load_end_file(path=".end")

ASPECT_CATEGORIES = [
    "cleanliness", "staff", "food", "location", "value", "noise",
    "amenities", "safety_security", "health", "billing", "room",
]

PROMPT_TEMPLATE = """You are analyzing a hotel guest review for a hotel operations team.

Review:
\"\"\"{review_text}\"\"\"

Return ONLY valid JSON with this exact structure:
{{
  "aspects": {{"<aspect>": "positive"|"negative"|"neutral", ...}},
  "overall_sentiment": "positive"|"negative"|"neutral",
  "escalate": true|false,
  "severity": "none"|"low"|"medium"|"high",
  "escalate_reason": "<short reason or null>"
}}

Rules:
- Only include aspects from this list that are actually discussed: {aspects}.
- Set escalate=true ONLY if the review describes a safety, health/pest,
  security, or billing-fraud issue (e.g. bed bugs, theft, broken locks,
  illness, being overcharged). Do not escalate for ordinary quality
  complaints (slow service, small room, bad food) alone.
- severity should reflect how urgent the issue is for hotel staff to act on.
"""

# Disable stub/demo mode by default so pasted-in reviews call the live API
# Set the environment variable USE_STUB=true to force demo mode.
USE_STUB = os.environ.get("USE_STUB", "false").lower() == "true"


def _extract_json(text: str) -> str:
    """Return the first JSON object found in a string (simple sanitizer)."""
    if not text or not isinstance(text, str):
        raise ValueError("Empty model response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def _send_prompt(prompt_text: str) -> str:
    """Try supported providers in order and return the raw text response.

    Order: Gemini -> OpenRouter -> OpenAI. Raise ValueError if none succeed.
    """
    # 1) Try Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=prompt_text,
            )
            return resp.text
        except Exception:
            # fall through to other providers
            pass

    # 2) Try OpenRouter (https://api.openrouter.ai)
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            import requests

            model = os.environ.get("OPENROUTER_MODEL", "gpt-4o-mini")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_text}],
            }
            headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
            r = requests.post(
                "https://api.openrouter.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 3) Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            openai.api_key = os.environ["OPENAI_API_KEY"]
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

    raise ValueError(
        "No usable LLM key found. Set GEMINI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY, or enable USE_STUB=true."
    )


def analyze_review(review_text: str, review_id: int = None) -> dict:
    """Return the aspect/escalation JSON for one review."""
    if USE_STUB:
        from llm_aspect_data import LLM_OUTPUTS

        if review_id is None or review_id not in LLM_OUTPUTS:
            raise ValueError(
                "No stub output for this review_id. Set USE_STUB=False and "
                "configure GEMINI_API_KEY to run this on new/live reviews."
            )
        return LLM_OUTPUTS[review_id]

    prompt = PROMPT_TEMPLATE.format(
        review_text=review_text, aspects=", ".join(ASPECT_CATEGORIES)
    )

    raw = _send_prompt(prompt)
    parsed = _extract_json(raw)
    return json.loads(parsed)


def draft_response(review_text: str, analysis: dict, review_id: int = None) -> str:
    """Draft a management response for an escalated review."""
    if USE_STUB:
        from llm_aspect_data import DRAFT_RESPONSES
        return DRAFT_RESPONSES.get(
            review_id,
            "No stub draft available for a pasted-in review -- set "
            "USE_STUB=False with a GEMINI_API_KEY to generate one live.",
        )

    prompt = f"""Draft a short, professional management response (3-4 sentences) to this
hotel review, which raised the following issue: {analysis.get('escalate_reason')}.
Acknowledge the specific issue, apologize, and state that it is being
escalated internally. Do not invent details not in the review.

Review: \"\"\"{review_text}\"\"\""""

    try:
        return _send_prompt(prompt)
    except ValueError:
        return "No draft available because no LLM key is configured. Set GEMINI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY, or enable USE_STUB=true."

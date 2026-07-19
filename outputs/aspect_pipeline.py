"""
Aspect-based sentiment + escalation pipeline.

This is the real pipeline your Streamlit app calls. It sends each review to
Claude (Anthropic) with the prompt template below and parses the JSON it
returns. To run this for real, set ANTHROPIC_API_KEY as an environment
variable and install the anthropic package (`pip install anthropic`).

In this sandbox there was no API key available, so llm_aspect_data.py
contains pre-computed outputs (produced by Claude reasoning directly over
the same 35 reviews) standing in for what this function would return live.
Swap USE_STUB = False once you have a key and it calls the real API.
"""
import json
import os

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

USE_STUB = True  # flip to False once ANTHROPIC_API_KEY is configured


def analyze_review(review_text: str, review_id: int = None) -> dict:
    """Return the aspect/escalation JSON for one review."""
    if USE_STUB:
        from llm_aspect_data import LLM_OUTPUTS
        if review_id is None or review_id not in LLM_OUTPUTS:
            raise ValueError(
                "No stub output for this review_id. Set USE_STUB=False and "
                "configure ANTHROPIC_API_KEY to run this on new/live reviews."
            )
        return LLM_OUTPUTS[review_id]

    # --- Real API call path ---
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = PROMPT_TEMPLATE.format(
        review_text=review_text, aspects=", ".join(ASPECT_CATEGORIES)
    )
    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(msg.content[0].text)


def draft_response(review_text: str, analysis: dict, review_id: int = None) -> str:
    """Draft a management response for an escalated review."""
    if USE_STUB:
        from llm_aspect_data import DRAFT_RESPONSES
        return DRAFT_RESPONSES.get(
            review_id,
            "No stub draft available for a pasted-in review -- set "
            "USE_STUB=False with an ANTHROPIC_API_KEY to generate one live.",
        )

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"""Draft a short, professional management response (3-4 sentences) to this
hotel review, which raised the following issue: {analysis.get('escalate_reason')}.
Acknowledge the specific issue, apologize, and state that it is being
escalated internally. Do not invent details not in the review.

Review: \"\"\"{review_text}\"\"\""""
    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

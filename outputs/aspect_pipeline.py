"""
Aspect-based sentiment + escalation pipeline.

This is the real pipeline your Streamlit app calls. It sends each review to
an LLM (Gemini or OpenRouter, chosen in the app's sidebar) with the prompt
template below and parses the JSON it returns.

In this sandbox there was no API key available, so llm_aspect_data.py
contains pre-computed outputs (produced by Claude reasoning directly over
the same 35 reviews) standing in for what this function would return live.
Set USE_STUB=true to fall back to that demo data instead of calling a live API.
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

# Providers this pipeline knows how to call, and the default model used for
# each unless the caller (the Streamlit sidebar) overrides it.
#
# Gemini's list below is hardcoded from ai.google.dev/gemini-api/docs/models
# (checked live), limited to stable + preview models that are not marked
# shut down. OpenRouter's list is fetched live at runtime instead (see
# list_openrouter_free_models below) because its free-tier lineup changes
# often; the entries here are just a handful of well-known paid defaults.
PROVIDERS = {
    "gemini": {
        "label": "Gemini",
        "key_env": "GEMINI_API_KEY",
        "models": [
            "gemini-3.6-flash",       # latest stable, best speed/intelligence balance
            "gemini-3.5-flash",       # stable, near-Pro intelligence
            "gemini-3.5-flash-lite",  # stable, lite/cost-efficient
            "gemini-3.1-flash-lite",  # stable, lite
            "gemini-3.1-pro-preview", # preview, most capable Gemini 3 Pro
            "gemini-3-flash-preview", # preview
            "gemini-2.5-pro",         # previous gen, still supported
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
    },
    "openrouter": {
        "label": "OpenRouter",
        "key_env": "OPENROUTER_API_KEY",
        "models": [
            "openai/gpt-5.6-sol",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-haiku-latest",
            "x-ai/grok-4.5",
            "google/gemini-3.5-flash",
        ],
    },
}

# Static fallback if the live OpenRouter catalog fetch fails (no network,
# rate limited, etc). This is NOT a complete or authoritative snapshot of
# OpenRouter's free tier -- it's only what a partial, size-limited preview
# fetch happened to catch during development, biased toward newest-released
# models. Do not read this as "these are the only free models"; DeepSeek,
# Llama, Gemma, or free Mistral variants may well exist and simply weren't
# in that partial fetch. list_openrouter_free_models() below is the real
# source of truth: it queries OpenRouter directly at runtime (from wherever
# the app is actually running, with no size cap) and returns whatever is
# free at that moment. This list is only used if that live call fails.
_OPENROUTER_FREE_FALLBACK = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "tencent/hy3:free",
    "nousresearch/hermes-4-405b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
]


def list_openrouter_free_models() -> list:
    """Fetch OpenRouter's live model catalog and return the current free ones.

    OpenRouter adds/removes free (":free", $0 prompt+completion) models
    often, so this hits the public, unauthenticated /models endpoint at
    call time rather than relying on a hardcoded list. Falls back to a
    small static snapshot if the request fails.
    """
    try:
        import requests

        r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        free_ids = []
        for m in data:
            mid = m.get("id", "")
            pricing = m.get("pricing", {}) or {}
            is_free = mid.endswith(":free") or (
                str(pricing.get("prompt")) in ("0", "0.0") and
                str(pricing.get("completion")) in ("0", "0.0")
            )
            if is_free and mid:
                free_ids.append(mid)
        free_ids = sorted(set(free_ids))
        return free_ids or _OPENROUTER_FREE_FALLBACK
    except Exception:
        return _OPENROUTER_FREE_FALLBACK


def get_models_for_provider(provider: str) -> list:
    """Return the full model dropdown list for a provider: Gemini's static
    list, or OpenRouter's paid defaults plus its live-fetched free models."""
    provider = (provider or "").lower()
    if provider == "gemini":
        return list(PROVIDERS["gemini"]["models"])
    if provider == "openrouter":
        return list(PROVIDERS["openrouter"]["models"]) + list_openrouter_free_models()
    return []

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


def _call_gemini(prompt_text: str, api_key: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt_text)
    return resp.text


def _call_openrouter(prompt_text: str, api_key: str, model: str) -> str:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _send_prompt(prompt_text: str, provider: str = None, model: str = None,
                  api_key: str = None) -> str:
    """Send a prompt to the given provider and return the raw text response.

    If `provider` is given, only that provider is tried (using `api_key` if
    supplied, else the provider's environment variable). If `provider` is
    None, Gemini then OpenRouter are tried in order using whatever keys are
    set in the environment. Raises ValueError if no usable key/provider
    is available or the call fails.
    """
    candidates = [provider] if provider else ["gemini", "openrouter"]

    last_error = None
    for prov in candidates:
        prov = (prov or "").lower()
        if prov not in PROVIDERS:
            continue
        key = api_key if (provider and api_key) else os.environ.get(PROVIDERS[prov]["key_env"])
        if not key:
            last_error = f"No API key set for {PROVIDERS[prov]['label']}."
            continue
        chosen_model = model or PROVIDERS[prov]["models"][0]
        try:
            if prov == "gemini":
                return _call_gemini(prompt_text, key, chosen_model)
            if prov == "openrouter":
                return _call_openrouter(prompt_text, key, chosen_model)
        except Exception as e:
            last_error = f"{PROVIDERS[prov]['label']} call failed: {e}"
            if provider:
                # Caller asked for this specific provider; don't silently
                # fall through to another one.
                raise ValueError(last_error) from e
            continue

    raise ValueError(
        last_error or "No usable LLM key found. Set GEMINI_API_KEY or OPENROUTER_API_KEY, or enable USE_STUB=true."
    )


def test_connection(provider: str, api_key: str, model: str = None) -> dict:
    """Send a trivial prompt to the given provider/key and report success.

    Returns {"ok": True, "detail": "..."} or {"ok": False, "detail": "..."}.
    Never raises; safe to call directly from a UI button handler.
    """
    provider = (provider or "").lower()
    if provider not in PROVIDERS:
        return {"ok": False, "detail": f"Unknown provider '{provider}'."}
    if not api_key:
        return {"ok": False, "detail": f"No {PROVIDERS[provider]['label']} API key entered."}

    chosen_model = model or PROVIDERS[provider]["models"][0]
    try:
        text = _send_prompt(
            "Reply with exactly one word: OK",
            provider=provider,
            model=chosen_model,
            api_key=api_key,
        )
        snippet = (text or "").strip().replace("\n", " ")[:80]
        return {
            "ok": True,
            "detail": f"{PROVIDERS[provider]['label']} ({chosen_model}) responded: \"{snippet}\"",
        }
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def analyze_review(review_text: str, review_id: int = None, provider: str = None,
                    model: str = None, api_key: str = None) -> dict:
    """Return the aspect/escalation JSON for one review."""
    if USE_STUB:
        from llm_aspect_data import LLM_OUTPUTS

        if review_id is None or review_id not in LLM_OUTPUTS:
            raise ValueError(
                "No stub output for this review_id. Set USE_STUB=False and "
                "configure an API key to run this on new/live reviews."
            )
        return LLM_OUTPUTS[review_id]

    prompt = PROMPT_TEMPLATE.format(
        review_text=review_text, aspects=", ".join(ASPECT_CATEGORIES)
    )

    raw = _send_prompt(prompt, provider=provider, model=model, api_key=api_key)
    parsed = _extract_json(raw)
    return json.loads(parsed)


def draft_response(review_text: str, analysis: dict, review_id: int = None,
                    provider: str = None, model: str = None, api_key: str = None) -> str:
    """Draft a management response for an escalated review."""
    if USE_STUB:
        from llm_aspect_data import DRAFT_RESPONSES
        return DRAFT_RESPONSES.get(
            review_id,
            "No stub draft available for a pasted-in review -- set "
            "USE_STUB=False and configure an API key to generate one live.",
        )

    prompt = f"""Draft a short, professional management response (3-4 sentences) to this
hotel review, which raised the following issue: {analysis.get('escalate_reason')}.
Acknowledge the specific issue, apologize, and state that it is being
escalated internally. Do not invent details not in the review.

Review: \"\"\"{review_text}\"\"\""""

    try:
        return _send_prompt(prompt, provider=provider, model=model, api_key=api_key)
    except ValueError:
        return "No draft available because no LLM key is configured. Set GEMINI_API_KEY or OPENROUTER_API_KEY, or enable USE_STUB=true."

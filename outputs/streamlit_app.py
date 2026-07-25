"""
Guest Review Triage -- aspect-based sentiment + escalation prototype.

Run with:  streamlit run streamlit_app.py

What it does:
- Paste/select a hotel guest review.
- Runs it through the aspect-based sentiment + escalation pipeline
  (aspect_pipeline.py -- calls Gemini or OpenRouter depending on what you
  pick in the sidebar, otherwise uses the pre-computed demo outputs in
  llm_aspect_data.py).
- Shows per-aspect sentiment, overall sentiment, an escalation flag with
  severity, and (for escalated reviews) a draft management response.
- Also shows what a plain sentiment classifier (VADER) would have said,
  to make the value-add of the aspect-based approach visible side by side.
"""
import json
import os
import zipfile
import pandas as pd
import streamlit as st
import nltk

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
nltk.data.path.append(os.path.join(_HERE, "nltk_data"))
from nltk.sentiment import SentimentIntensityAnalyzer

LEXICON_DIR = os.path.join(_HERE, 'nltk_data', 'sentiment', 'vader_lexicon')
LEXICON_FILE = os.path.join(LEXICON_DIR, 'vader_lexicon.txt')
LEXICON_ZIP = os.path.join(_HERE, 'nltk_data', 'sentiment', 'vader_lexicon.zip')

if os.path.exists(LEXICON_FILE) and not os.path.exists(LEXICON_ZIP):
    with zipfile.ZipFile(LEXICON_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(LEXICON_FILE, arcname='vader_lexicon/vader_lexicon.txt')

from aspect_pipeline import (
    analyze_review, draft_response, test_connection, PROVIDERS,
    get_models_for_provider, ASPECT_CATEGORIES,
)
from llm_aspect_data import LLM_OUTPUTS, DRAFT_RESPONSES

st.set_page_config(
    page_title="Guest Review Triage", layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Guest Review Triage")
st.caption(
    "Aspect-based sentiment + escalation flagging for hotel guest reviews. "
    "Prototype for AI-Powered Process Automation assignment."
)

# ---------------------------------------------------------------------------
# Sidebar: LLM provider + API key configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("LLM settings")
    st.caption(
        "Keys are kept only in this app's memory for the current session. "
        "They are never written to disk or logged."
    )

    provider_label_to_key = {v["label"]: k for k, v in PROVIDERS.items()}
    provider_choice_label = st.selectbox(
        "Model provider", list(provider_label_to_key.keys()), index=0
    )
    provider_choice = provider_label_to_key[provider_choice_label]

    @st.cache_data(ttl=3600)
    def _cached_models(provider: str):
        return get_models_for_provider(provider)

    with st.spinner("Loading model list..."):
        model_options = _cached_models(provider_choice)
    model_choice = st.selectbox(
        "Model", model_options, index=0,
        key=f"model_{provider_choice}",
        help="OpenRouter's free (\":free\") models are fetched live and change often.",
    )

    st.divider()
    gemini_key = st.text_input(
        "Gemini API key", type="password",
        value=st.session_state.get("gemini_key", ""),
        key="gemini_key_input",
    )
    openrouter_key = st.text_input(
        "OpenRouter API key", type="password",
        value=st.session_state.get("openrouter_key", ""),
        key="openrouter_key_input",
    )
    st.session_state["gemini_key"] = gemini_key
    st.session_state["openrouter_key"] = openrouter_key

    active_key = gemini_key if provider_choice == "gemini" else openrouter_key

    st.divider()
    if st.button("Test connection"):
        if not active_key:
            st.error(f"Enter a {provider_choice_label} API key above first.")
        else:
            with st.spinner(f"Pinging {provider_choice_label}..."):
                result = test_connection(provider_choice, active_key, model_choice)
            if result["ok"]:
                st.success(result["detail"])
            else:
                st.error(result["detail"])

sia = SentimentIntensityAnalyzer()


def vader_label(text):
    c = sia.polarity_scores(text)["compound"]
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


@st.cache_data
def load_sample_reviews():
    df = pd.read_csv(os.path.join(DATA_DIR, "llm_subset.csv"))
    return df


samples = load_sample_reviews()

st.subheader("1. Choose a review")
mode = st.radio("Input", ["Pick a sample review", "Paste my own review"], horizontal=True)

review_id = None
if mode == "Pick a sample review":
    idx = st.selectbox(
        "Sample reviews (from the TripAdvisor hotel reviews dataset)",
        options=samples.index,
        format_func=lambda i: f"#{i} (rating {samples.loc[i, 'Rating']}/5) -- "
        + samples.loc[i, "Review"][:80] + "...",
    )
    review_text = samples.loc[idx, "Review"]
    review_id = idx
else:
    review_text = st.text_area("Paste a review here", height=150)

st.text_area("Review text", review_text, height=150, disabled=True, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Session state: running comparison log, one entry per "Analyze review" click
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "current_id" not in st.session_state:
    st.session_state.current_id = None


def _entry_by_id(entry_id):
    for e in st.session_state.history:
        if e["id"] == entry_id:
            return e
    return None


def _effective_escalate(entry):
    """True escalation status after applying any manager override."""
    if entry["manual_override"] == "escalate":
        return True
    if entry["manual_override"] == "clear":
        return False
    return entry["model_escalate"]


if st.button("Analyze review", type="primary") and review_text.strip():
    with st.spinner("Analyzing..."):
        try:
            result = analyze_review(
                review_text, review_id=review_id,
                provider=provider_choice, model=model_choice, api_key=active_key,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    entry = {
        "id": st.session_state.next_id,
        "review_label": f"#{review_id}" if review_id is not None else f"Pasted: {review_text[:30]}...",
        "review_text": review_text,
        "review_id": review_id,
        "provider": provider_choice_label,
        "model": model_choice,
        "aspects": result["aspects"],
        "overall_sentiment": result["overall_sentiment"],
        "model_escalate": result["escalate"],
        "severity": result["severity"],
        "escalate_reason": result.get("escalate_reason"),
        "manual_override": None,
        "manual_reason": "",
        "vader_label": vader_label(review_text),
        "draft": None,
    }
    st.session_state.history.append(entry)
    st.session_state.next_id += 1
    st.session_state.current_id = entry["id"]

current = _entry_by_id(st.session_state.current_id) if st.session_state.current_id else None

if current is not None:
    st.subheader("2. Aspect-based sentiment")
    aspect_cols = st.columns(3)
    for i, (aspect, sentiment) in enumerate(current["aspects"].items()):
        color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[sentiment]
        with aspect_cols[i % 3]:
            st.metric(aspect.replace("_", " ").title(), f"{color} {sentiment}")

    st.subheader("3. Overall sentiment & escalation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall sentiment", current["overall_sentiment"].title())
    c2.metric("Model escalation call", "Yes" if current["model_escalate"] else "No")
    c3.metric("Severity", current["severity"].title())

    # -----------------------------------------------------------------
    # Manager override: two-way -- can force-escalate a miss, or clear
    # a flag they judge to be a false positive.
    # -----------------------------------------------------------------
    st.subheader("4. Manager review")
    effective = _effective_escalate(current)
    if current["manual_override"] == "escalate":
        st.info("Manually escalated by manager (model said No).")
    elif current["manual_override"] == "clear":
        st.info("Manually cleared by manager (model said Yes, judged not urgent).")

    override_cols = st.columns([1, 1, 2])
    with override_cols[0]:
        if not current["model_escalate"] and current["manual_override"] != "escalate":
            if st.button("Escalate manually", key=f"escalate_{current['id']}"):
                current["manual_override"] = "escalate"
                current["draft"] = None
                st.rerun()
        elif current["manual_override"] == "escalate":
            if st.button("Undo manual escalation", key=f"undo_escalate_{current['id']}"):
                current["manual_override"] = None
                current["draft"] = None
                st.rerun()
    with override_cols[1]:
        if current["model_escalate"] and current["manual_override"] != "clear":
            if st.button("Clear (false positive)", key=f"clear_{current['id']}"):
                current["manual_override"] = "clear"
                current["draft"] = None
                st.rerun()
        elif current["manual_override"] == "clear":
            if st.button("Undo clear", key=f"undo_clear_{current['id']}"):
                current["manual_override"] = None
                current["draft"] = None
                st.rerun()

    if current["manual_override"] == "escalate":
        current["manual_reason"] = st.text_input(
            "Manager's reason for escalating (used in the draft response)",
            value=current["manual_reason"],
            key=f"reason_{current['id']}",
        )

    # -----------------------------------------------------------------
    # Draft response: generated only when the review is, right now,
    # effectively escalated -- by the model or by the manager.
    # -----------------------------------------------------------------
    if effective:
        st.subheader("5. Draft management response")
        if current["draft"] is None:
            draft_analysis = dict(current)
            if current["manual_override"] == "escalate" and current["manual_reason"].strip():
                draft_analysis["escalate_reason"] = current["manual_reason"].strip()
            elif current["manual_override"] == "escalate" and not current["escalate_reason"]:
                draft_analysis["escalate_reason"] = "Flagged by manager for escalation (see review)."
            with st.spinner("Drafting response..."):
                current["draft"] = draft_response(
                    current["review_text"], draft_analysis, review_id=current["review_id"],
                    provider=provider_choice, model=model_choice, api_key=active_key,
                )
        st.text_area(
            "Editable draft (human reviews before sending)",
            current["draft"], height=140, key=f"draft_box_{current['id']}",
        )
        if st.button("Regenerate draft", key=f"regen_{current['id']}"):
            current["draft"] = None
            st.rerun()
    else:
        st.success("No escalation needed -- routine feedback. No draft is generated.")

    # -----------------------------------------------------------------
    # VADER, shown in its own box, compared against the model's call.
    # -----------------------------------------------------------------
    st.subheader("6. VADER baseline (plain sentiment classifier)")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.metric("Model overall sentiment", current["overall_sentiment"].title())
    with vc2:
        st.metric("VADER overall sentiment", current["vader_label"].title())
    if current["overall_sentiment"].lower() == current["vader_label"].lower():
        st.caption("VADER agrees with the model's overall sentiment.")
    else:
        st.caption(
            "VADER disagrees with the model's overall sentiment. "
            "VADER has no aspects and no escalation logic -- it's a single "
            "lexicon-based score over the whole review, shown here only as a floor baseline."
        )

# ---------------------------------------------------------------------------
# Running comparison table across every analysis run this session
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Session comparison table")
if not st.session_state.history:
    st.caption("Run \"Analyze review\" to start building a comparison table here.")
else:
    rows = []
    for e in st.session_state.history:
        eff = _effective_escalate(e)
        escalate_label = "Yes" if eff else "No"
        if e["manual_override"] == "escalate":
            escalate_label += " (manual)"
        elif e["manual_override"] == "clear":
            escalate_label += " (cleared)"
        rows.append({
            "Review": e["review_label"],
            "Provider": e["provider"],
            "Model": e["model"],
            "Food": e["aspects"].get("food", "-"),
            "Staff": e["aspects"].get("staff", "-"),
            "Health": e["aspects"].get("health", "-"),
            "Overall sentiment": e["overall_sentiment"],
            "Escalate": escalate_label,
            "Severity": e["severity"],
            "VADER sentiment": e["vader_label"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if st.button("Clear comparison table"):
        st.session_state.history = []
        st.session_state.current_id = None
        st.rerun()

st.divider()
st.caption(
    "Aspects tracked: " + ", ".join(ASPECT_CATEGORIES) +
    ". Escalation is triggered only for safety, health/pest, security, or billing-fraud issues, "
    "or by manager override above."
)
st.caption("Created by Group C6")

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

    st.subheader("2. Aspect-based sentiment")
    aspect_cols = st.columns(3)
    for i, (aspect, sentiment) in enumerate(result["aspects"].items()):
        color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[sentiment]
        with aspect_cols[i % 3]:
            st.metric(aspect.replace("_", " ").title(), f"{color} {sentiment}")

    st.subheader("3. Overall sentiment & escalation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall sentiment", result["overall_sentiment"].title())
    c2.metric("Escalate?", "Yes" if result["escalate"] else "No")
    c3.metric("Severity", result["severity"].title())

    if result["escalate"]:
        st.warning(f"**Why flagged:** {result['escalate_reason']}")
        st.subheader("4. Draft management response")
        draft = draft_response(
            review_text, result, review_id=review_id,
            provider=provider_choice, model=model_choice, api_key=active_key,
        )
        st.text_area("Editable draft (human reviews before sending)", draft, height=140)
    else:
        st.success("No escalation needed -- routine feedback.")

    with st.expander("Compare: what a plain sentiment classifier (VADER) would say"):
        st.write(f"VADER overall sentiment: **{vader_label(review_text)}** (no aspects, no escalation flag)")

st.divider()
st.caption(
    "Aspects tracked: " + ", ".join(ASPECT_CATEGORIES) +
    ". Escalation is triggered only for safety, health/pest, security, or billing-fraud issues."
)
st.caption("Created by Group C6")

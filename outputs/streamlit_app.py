"""
Guest Review Triage -- aspect-based sentiment + escalation prototype.

Run with:  streamlit run streamlit_app.py

What it does:
- Paste/select a hotel guest review.
- Runs it through the aspect-based sentiment + escalation pipeline
  (aspect_pipeline.py -- calls Claude if ANTHROPIC_API_KEY is set,
  otherwise uses the pre-computed demo outputs in llm_aspect_data.py).
- Shows per-aspect sentiment, overall sentiment, an escalation flag with
  severity, and (for escalated reviews) a draft management response.
- Also shows what a plain sentiment classifier (VADER) would have said,
  to make the value-add of the aspect-based approach visible side by side.
"""
import json
import os
import pandas as pd
import streamlit as st
import nltk

_HERE = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.append(os.path.join(_HERE, "nltk_data"))
from nltk.sentiment import SentimentIntensityAnalyzer

from aspect_pipeline import analyze_review, draft_response, ASPECT_CATEGORIES
from llm_aspect_data import LLM_OUTPUTS, DRAFT_RESPONSES

st.set_page_config(page_title="Guest Review Triage", layout="centered")
st.title("Guest Review Triage")
st.caption(
    "Aspect-based sentiment + escalation flagging for hotel guest reviews. "
    "Prototype for AI-Powered Process Automation assignment."
)

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
    df = pd.read_csv("data/llm_subset.csv")
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
            result = analyze_review(review_text, review_id=review_id)
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
        draft = draft_response(review_text, result, review_id=review_id)
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

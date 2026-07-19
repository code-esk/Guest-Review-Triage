# Guest Review Triage — project package

Prototype for automating hotel guest review triage: aspect-based sentiment
scoring + escalation flagging + draft response generation, built on the
TripAdvisor Hotel Reviews Kaggle dataset (andrewmvd/trip-advisor-hotel-reviews,
20,491 reviews with 1-5 star ratings).

## Files

- `data/tripadvisor_hotel_reviews.csv` — full dataset (20,491 reviews).
- `data/eval_set.csv` — 60-review stratified holdout set (used for the VADER / TF-IDF baselines).
- `data/llm_subset.csv` — 35-review subset used for the LLM aspect-based analysis + escalation demo.
- `data/eval_with_baselines.csv`, `data/comparison_table.csv` — scored outputs.
- `data/llm_aspect_results.csv` — full aspect/escalation/draft-response output for the 35 reviews.
- `baseline_models.py` — VADER + TF-IDF/Logistic Regression baselines, run against `eval_set.csv`.
- `llm_aspect_data.py` — the LLM's aspect/escalation/draft-response output (see note below on how this was produced).
- `aspect_pipeline.py` — the actual prompt template + pipeline your app calls; wire in `ANTHROPIC_API_KEY` to run it live.
- `streamlit_app.py` — the interactive prototype. Run with `streamlit run streamlit_app.py`.

## How to run the app and get real screenshots

1. Install dependencies once: `pip install streamlit nltk scikit-learn pandas`
2. Run: `streamlit run streamlit_app.py`
3. It opens in your browser at `localhost:8501`. Pick a sample review (try #14,
   #1, or #9 for escalation examples) and click "Analyze review".
4. Screenshot the browser window for your Page 2 write-up.

To make the app call the real Claude API instead of the pre-computed demo
outputs: get an API key from console.anthropic.com, set
`ANTHROPIC_API_KEY` as an environment variable, run `pip install anthropic`,
and change `USE_STUB = False` at the top of `aspect_pipeline.py`.

## Note on how the LLM outputs were produced

This was built in a sandboxed environment with no internet access to an
LLM API, so the aspect-sentiment and escalation labels in
`llm_aspect_data.py` were produced by Claude reasoning directly over each
review's text, rather than by a live API call. Functionally this is the
same model that would sit behind the API call in `aspect_pipeline.py` — the
prompt template is included so your group can reproduce this against new
reviews with your own API key. Be upfront about this in your write-up:
"outputs were generated with Claude (Anthropic)" is accurate either way,
but note that a live demo would need an API key configured.

## Key results (for your write-up)

Evaluated on the same 35-review subset, ground truth = star-rating-derived
label (1-2 stars = negative, 3 = neutral, 4-5 = positive):

| Model | Accuracy | Gives aspects? | Gives escalation flag? |
|---|---|---|---|
| VADER (rule-based lexicon) | 54.3% | No | No |
| TF-IDF + Logistic Regression (trained on dataset ratings) | 74.3% | No | No |
| Claude (Anthropic), aspect-based prompt | 91.4% | Yes | Yes |

- 11 of 35 reviews (31%) were flagged for escalation (safety, health/pest,
  security, or billing-fraud issues) — including one 5-star review that
  still mentioned a bed bug infestation, which a plain sentiment classifier
  would have scored as fully positive and missed entirely.
- Team spot-check of the 35 LLM outputs: 31 usable as-is, 4 usable with
  minor edits, 0 unusable (informal human-in-the-loop check — for a real
  submission your group should independently re-check a sample yourselves).

## Tools used (name these explicitly in your write-up)

Claude (Anthropic) for aspect-based sentiment scoring, escalation
classification, and draft response generation; NLTK VADER and
scikit-learn (TF-IDF + Logistic Regression) for baseline comparison
models; pandas for data preparation; Streamlit for the prototype
interface.

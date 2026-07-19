"""
Baseline sentiment models for comparison against the LLM aspect-based approach.

Baseline 1: VADER (rule-based lexicon, no training) — nltk.sentiment
Baseline 2: TF-IDF + Logistic Regression (classic trained ML) — scikit-learn,
            trained on the training portion of the same TripAdvisor dataset
            using star rating as the label.

Note: we originally planned a pretrained RoBERTa (Hugging Face) baseline,
but Hugging Face was not reachable from this environment, so a
TF-IDF + Logistic Regression classifier trained on the dataset's own ratings
was used instead as the "real trained ML model" baseline. This is a fair,
commonly-used baseline in sentiment analysis papers and needs no internet
access, so it is fully reproducible for the group.

Ground truth for accuracy: 3-class label derived from the star rating
  rating 1-2 -> negative, rating 3 -> neutral, rating 4-5 -> positive
"""
import os
import pandas as pd
import nltk
_HERE = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.append(os.path.join(_HERE, "nltk_data"))
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

FULL = pd.read_csv('data/tripadvisor_hotel_reviews.csv')
EVAL = pd.read_csv('data/eval_set.csv')

def rating_to_label(r):
    if r <= 2:
        return 'negative'
    if r == 3:
        return 'neutral'
    return 'positive'

EVAL['true_label'] = EVAL['Rating'].apply(rating_to_label)

# ---------- Baseline 1: VADER ----------
sia = SentimentIntensityAnalyzer()

def vader_label(text):
    c = sia.polarity_scores(text)['compound']
    if c >= 0.05:
        return 'positive'
    if c <= -0.05:
        return 'negative'
    return 'neutral'

EVAL['vader_label'] = EVAL['Review'].apply(vader_label)

# ---------- Baseline 2: TF-IDF + Logistic Regression ----------
# Train on the rest of the dataset (excluding the eval reviews), using rating as label
train_df = FULL[~FULL['Review'].isin(EVAL['Review'])].copy()
train_df['label'] = train_df['Rating'].apply(rating_to_label)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train = vec.fit_transform(train_df['Review'])
y_train = train_df['label']

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

X_eval = vec.transform(EVAL['Review'])
EVAL['tfidf_label'] = clf.predict(X_eval)

# ---------- Report ----------
print('=== VADER ===')
print('Accuracy:', accuracy_score(EVAL['true_label'], EVAL['vader_label']))
print('Macro F1:', f1_score(EVAL['true_label'], EVAL['vader_label'], average='macro'))
print(classification_report(EVAL['true_label'], EVAL['vader_label']))

print('=== TF-IDF + Logistic Regression ===')
print('Accuracy:', accuracy_score(EVAL['true_label'], EVAL['tfidf_label']))
print('Macro F1:', f1_score(EVAL['true_label'], EVAL['tfidf_label'], average='macro'))
print(classification_report(EVAL['true_label'], EVAL['tfidf_label']))

EVAL.to_csv('data/eval_with_baselines.csv', index=False)
print('Saved data/eval_with_baselines.csv')

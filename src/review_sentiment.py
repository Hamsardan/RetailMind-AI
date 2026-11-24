"""Day 12: classical Amazon review sentiment analysis."""
from pathlib import Path
from html import unescape
import csv
import random
import re

import joblib
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'review_sentiment_day12.joblib'
LABELS = {0: 'Negative', 1: 'Positive'}
TOKENIZER = RegexpTokenizer(r'[a-z]+')
# Negation matters for sentiment: do not remove it as a stopword.
STOP_WORDS = (ENGLISH_STOP_WORDS | {'did', 'does', 'doing', 'had', 'having'}) - {
    'no', 'not', 'never', 'nor', 'without', 'cannot'
}


def tokenize_review(text):
    """Lowercase, remove HTML/URLs/noise, tokenize and filter stopwords."""
    text = unescape(text).lower().replace('’', "'")
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = text.replace(r'\n', ' ').replace(r'\r', ' ')
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"\bshan't\b", 'shall not', text)
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r"\b([a-z]+)'(?:s|re|ve|ll|d|m)\b", r'\1', text)
    return [word for word in TOKENIZER.tokenize(text)
            if len(word) > 1 and word not in STOP_WORDS]


def clean_review(text):
    return ' '.join(tokenize_review(text))


def load_amazon_reviews(csv_path, per_class, seed=42):
    """Sample a headerless original polarity CSV (1=negative, 2=positive).

    Reservoir sampling scans every row but keeps only per_class reviews per
    label in memory. The official train and test files are sampled separately.
    """
    if per_class < 1:
        raise ValueError('per_class must be positive')
    rng = random.Random(seed)
    samples = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as source:
        for row in csv.reader(source):
            if not row:
                continue
            if len(row) != 3 or row[0].strip() not in {'1', '2'}:
                raise ValueError('Expected a headerless Amazon Polarity CSV: label,title,content (1 or 2).')
            label = int(row[0]) - 1
            text = ' '.join((row[1].strip(), row[2].strip())).strip()
            if not text:
                continue
            seen[label] += 1
            bucket = samples[label]
            if len(bucket) < per_class:
                bucket.append((text, label))
            else:
                index = rng.randrange(seen[label])
                if index < per_class:
                    bucket[index] = (text, label)
    if not all(samples.values()):
        raise ValueError(f'Both sentiment classes are required in {csv_path}')
    # Keep the actual sample balanced, even for a small custom CSV.
    count = min(map(len, samples.values()))
    records = samples[0][:count] + samples[1][:count]
    rng.shuffle(records)
    texts, labels = zip(*records)
    return list(texts), list(labels)


def build_sentiment_pipeline(max_features=30000, min_df=2):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_review, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), max_features=max_features,
            min_df=min_df, sublinear_tf=True,
        )),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ])


def informative_terms(model, top_n=15):
    """Return the strongest signed logistic-regression coefficients."""
    terms = model.named_steps['tfidf'].get_feature_names_out()
    classifier = model.named_steps['classifier']
    weights = classifier.coef_[0]
    order = weights.argsort()
    negative = [i for i in order if weights[i] < -1e-12][:top_n]
    positive = [i for i in reversed(order) if weights[i] > 1e-12][:top_n]
    return {
        'Negative': [(str(terms[i]), float(weights[i])) for i in negative],
        'Positive': [(str(terms[i]), float(weights[i])) for i in positive],
    }


def save_sentiment_model(model, model_path=MODEL_PATH):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def predict_reviews(reviews, model_path=MODEL_PATH):
    """Predict from one raw review or a list of raw reviews."""
    if isinstance(reviews, str):
        reviews = [reviews]
    else:
        reviews = list(reviews)
    if not reviews or any(not isinstance(r, str) or not tokenize_review(r) for r in reviews):
        raise ValueError('Provide nonempty reviews containing usable English words.')
    model = joblib.load(model_path)
    predictions = model.predict(reviews)
    probabilities = model.predict_proba(reviews)
    positive_index = list(model.classes_).index(1)
    return [
        {'review': review, 'sentiment': LABELS[int(label)],
         'positive_probability': float(prob[positive_index]),
         'confidence': float(max(prob))}
        for review, label, prob in zip(reviews, predictions, probabilities)
    ]

# Day 12 — NLP & Classical Text Processing

## Objective
Build a reusable review sentiment baseline for RetailMind AI.

## Dataset
Amazon Review Polarity (original CSV); sources: [Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews), [fast.ai mirror](https://course20.fast.ai/datasets), [Hugging Face](https://huggingface.co/datasets/fancyzhx/amazon_polarity). The CSV labels 1 and 2 are mapped to negative (0) and positive (1). Reviews include title and content. Use up to 10,000 training and 2,000 test examples per class, sampled with a fixed seed from the official splits. Exact sampled train/test text overlaps are excluded from testing.

## Method
- Lowercase, HTML/URL removal, contraction handling, NLTK regex tokenization and stopword removal with negation retained.
- TF-IDF unigrams and bigrams (up to 30,000 features; fitted on training only).
- Logistic regression for binary sentiment.
- Accuracy, precision, recall, F1, macro F1, ROC-AUC, classification report and confusion matrix.
- Inspect signed classifier coefficients for informative positive and negative terms.

## Results
Not measured yet on the real Amazon data. Running the notebook writes actual metrics to `models/review_sentiment_day12_metrics.json` and informative terms to `docs/day12_sentiment_terms.csv`.

## Reusable inference
Run the notebook to create `models/review_sentiment_day12.joblib`. From the repository root, use `from src.review_sentiment import predict_reviews` and pass a string or a list of raw reviews. The persisted pipeline includes fitted TF-IDF and logistic regression and references the reusable tokenizer in `src/review_sentiment.py`.

## Limitations
Binary sentiment only; neutral and mixed reviews are not separate labels. The dataset does not include product IDs, so informative terms are corpus-level associations rather than product-specific root causes. No pretrained embeddings or BERT are used.

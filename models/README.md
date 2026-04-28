# Place trained model artifacts in the subdirectories below.
#
# ── TF-IDF artifacts (models/tfidf/) ─────────────────────────────────────
# Train a sklearn pipeline and save with joblib:
#
#   from sklearn.feature_extraction.text import TfidfVectorizer
#   from sklearn.linear_model import LogisticRegression
#   from sklearn.preprocessing import LabelEncoder
#   import joblib
#
#   vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
#   X_train = vectorizer.fit_transform(train_texts)
#   model = LogisticRegression(max_iter=1000)
#   model.fit(X_train, train_labels)
#
#   joblib.dump(vectorizer, "models/tfidf/tfidf_vectorizer.pkl")
#   joblib.dump(model,      "models/tfidf/news_classifier.pkl")
#   # Optional label encoder:
#   joblib.dump(encoder,    "models/tfidf/label_encoder.pkl")
#
# ── BERT artifacts (models/bert/) ────────────────────────────────────────
# Fine-tune a HuggingFace model and save with save_pretrained():
#
#   model.save_pretrained("models/bert/bbc-bert-finetuned")
#   tokenizer.save_pretrained("models/bert/bbc-bert-finetuned")
#   # Then set: BERT_MODEL_NAME=bbc-bert-finetuned in .env
#
# This directory is git-ignored. Do NOT commit model files.

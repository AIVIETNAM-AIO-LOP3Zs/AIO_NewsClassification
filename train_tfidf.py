"""
Train a TF-IDF classifier for BBC News classification.

Run:
    python train_tfidf.py

Outputs:
    models/tfidf/news_classifier.pkl
    models/tfidf/tfidf_vectorizer.pkl
    models/tfidf/label_encoder.pkl
    models/tfidf/label_mapping.json
    models/tfidf/bbc_text_classifier_pipeline.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    required_columns = {"text", "category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"[ERROR] Missing required columns: {sorted(missing_columns)}")
        sys.exit(1)

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    if df.empty:
        print("[ERROR] Dataset is empty after cleaning.")
        sys.exit(1)

    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict[str, int]]:
    label_encoder = LabelEncoder()
    df = df.copy()
    df["label_id"] = label_encoder.fit_transform(df["category"])
    label_mapping = {
        label: int(idx) for idx, label in enumerate(label_encoder.classes_)
    }
    return df, label_encoder, label_mapping


def print_data_summary(df: pd.DataFrame, label_mapping: dict[str, int]) -> None:
    print(f"Total samples: {len(df)}")
    print(f"Duplicate text: {df.duplicated(subset=['text']).sum()}")
    print("\nLabel counts:")
    print(df["category"].value_counts().sort_index())
    print(f"\nLabel mapping: {label_mapping}")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label_id"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label_id"],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def train_and_select_model(
    x_train,
    y_train,
    x_val,
    y_val,
) -> tuple[str, object, pd.DataFrame]:
    candidates = {
        "Dummy majority baseline": DummyClassifier(strategy="most_frequent"),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
    }

    results = []
    trained_models = {}

    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        results.append(
            {
                "model": model_name,
                "val_accuracy": accuracy_score(y_val, val_pred),
                "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
                "val_weighted_f1": f1_score(y_val, val_pred, average="weighted"),
            }
        )
        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    best_model_name = str(results_df.iloc[0]["model"])
    return best_model_name, trained_models[best_model_name], results_df


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "bbc_clean.csv"
    output_dir = project_dir / "models" / "tfidf"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {data_path}")
    df = load_dataset(data_path)
    df, label_encoder, label_mapping = encode_labels(df)
    print_data_summary(df, label_mapping)

    train_df, val_df, test_df = split_data(df)
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    text_col = "clean_text" if "clean_text" in train_df.columns else "text"
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
    )

    print("\n[INFO] Fitting TF-IDF vectorizer...")
    x_train = tfidf.fit_transform(train_df[text_col])
    x_val = tfidf.transform(val_df[text_col])
    x_test = tfidf.transform(test_df[text_col])

    print(f"TF-IDF features: {len(tfidf.get_feature_names_out())}")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_val shape: {x_val.shape}")
    print(f"X_test shape: {x_test.shape}")

    print("\n[INFO] Training candidate models...")
    best_model_name, best_model, results_df = train_and_select_model(
        x_train,
        train_df["label_id"],
        x_val,
        val_df["label_id"],
    )

    print("\nValidation results:")
    print(results_df.to_string(index=False))

    test_pred = best_model.predict(x_test)
    target_names = list(label_encoder.classes_)
    print(f"\nBest model: {best_model_name}")
    print(f"Test accuracy: {accuracy_score(test_df['label_id'], test_pred):.4f}")
    print(f"Test macro F1: {f1_score(test_df['label_id'], test_pred, average='macro'):.4f}")
    print("\nClassification report:")
    print(classification_report(test_df["label_id"], test_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(test_df["label_id"], test_pred))

    print(f"\n[INFO] Saving artifacts to {output_dir}")
    joblib.dump(best_model, output_dir / "news_classifier.pkl")
    joblib.dump(tfidf, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(label_mapping, file, indent=4)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf.get_params())),
            ("model", best_model),
        ]
    )
    pipeline.fit(train_df[text_col], train_df["label_id"])

    with (output_dir / "bbc_text_classifier_pipeline.pkl").open("wb") as file:
        pickle.dump(pipeline, file)

    saved_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"[SUCCESS] Saved files: {saved_files}")
    print("[DONE] TF-IDF training completed.")


if __name__ == "__main__":
    main()

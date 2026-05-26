"""
BBC News Classification - TF-IDF Training Script (Production Version)
=====================================================================
Script này thực hiện workflow tối giản:
  load data -> label encode -> TF-IDF Vectorization -> train LinearSVC -> save model
Không thực hiện evaluate hay split dữ liệu theo yêu cầu của Leader.
Khi chạy xong, sẽ xuất ra bộ weight tại models/tfidf/
"""

import json
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "models" / "tfidf"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data từ thư mục data
    data_csv = data_dir / "bbc_clean.csv"
    print(f"[INFO] Loading cleaned data from {data_csv}")
    if not data_csv.exists():
        print(f"[ERROR] Dataset not found at {data_csv}")
        sys.exit(1)

    df = pd.read_csv(str(data_csv))
    if df.empty:
        print("[ERROR] Dataset is empty.")
        sys.exit(1)

    # 2. Label Encoding
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["category"])
    
    label_mapping = {
        label: int(idx) for idx, label in enumerate(label_encoder.classes_)
    }
    print(f"[INFO] Label mapping: {label_mapping}")

    # 3. TF-IDF Vectorization
    print("[INFO] Vectorizing and training LinearSVC model...")
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
    )
    
    X = tfidf.fit_transform(df[text_col])
    y = df["label_id"]

    # 4. Train best model (LinearSVC)
    model = LinearSVC(random_state=42)
    model.fit(X, y)
    print("[INFO] Model training complete.")

    # 5. Save Model Artifacts
    print(f"[INFO] Saving model artifacts to {output_dir}...")
    joblib.dump(tfidf, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(model, output_dir / "news_classifier.pkl")
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")

    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=4)

    # Also save the pipeline version
    print("[INFO] Saving pipeline version...")
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            norm="l2",
        )),
        ("model", LinearSVC(random_state=42)),
    ])
    final_pipeline.fit(df[text_col], y)

    pipeline_path = output_dir / "bbc_text_classifier_pipeline.pkl"
    joblib.dump(final_pipeline, pipeline_path)

    print(f"[SUCCESS] Saved successfully to: {output_dir}")
    print("[DONE] TF-IDF training completed successfully!")

if __name__ == "__main__":
    main()

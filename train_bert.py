"""
Fine-tune BERT for BBC News classification.

Run:
    python train_bert.py

Outputs:
    models/bert/bbc-bert-finetuned/

This matches the default .env values:
    BERT_MODEL_DIR=models/bert
    BERT_MODEL_NAME=bbc-bert-finetuned
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


RANDOM_STATE = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


class BBCDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer: BertTokenizer) -> None:
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    if "category" in df.columns:
        df = df.rename(columns={"category": "label"})

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"[ERROR] Missing required columns: {sorted(missing_columns)}")
        sys.exit(1)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    if df.empty:
        print("[ERROR] Dataset is empty after cleaning.")
        sys.exit(1)

    return df


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(
    model: BertForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, f1, all_preds, all_labels


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "bbc_clean.csv"
    output_dir = project_dir / "models" / "bert" / "bbc-bert-finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {data_path}")
    df = load_dataset(data_path)
    print(f"Cleaned data shape: {df.shape}")
    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index())

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"\nLabel mapping: {label2id}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print(f"\n[INFO] Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BBCDataset(train_df["text"], train_df["label_id"], tokenizer)
    val_dataset = BBCDataset(val_df["text"], val_df["label_id"], tokenizer)
    test_dataset = BBCDataset(test_df["text"], test_df["label_id"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = resolve_device()
    print(f"\nUsing device: {device}")

    print(f"\n[INFO] Loading model: {MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    print("\n[INFO] Starting fine-tuning...")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in progress_bar:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val accuracy: {val_acc:.4f}")
        print(f"Val F1 macro: {val_f1:.4f}")

    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model,
        test_loader,
        device,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1 macro: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))

    print(f"\n[INFO] Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(label2id, file, indent=4)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    saved_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"[SUCCESS] Saved files: {saved_files}")
    print("[DONE] BERT fine-tuning completed.")


if __name__ == "__main__":
    main()

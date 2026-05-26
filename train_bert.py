"""
Fine-tune BERT for BBC News classification (Production Version)
==============================================================
Script này thực hiện fine-tuning BERT trực tiếp trên toàn bộ tập dữ liệu.
Không thực hiện evaluate hay split dữ liệu theo yêu cầu của Leader.
Khi chạy xong, sẽ xuất ra bộ weight tại models/bert/bbc-bert-finetuned/
"""

import json
import sys
from pathlib import Path
import pandas as pd
import torch
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
    return df

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "bbc_clean.csv"
    output_dir = project_dir / "models" / "bert" / "bbc-bert-finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {data_path}")
    df = load_dataset(data_path)
    print(f"Cleaned data shape: {df.shape}")

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"\nLabel mapping: {label2id}")

    print(f"\n[INFO] Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Train trên toàn bộ dataset
    dataset = BBCDataset(df["text"], df["label_id"], tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    print("\n[INFO] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

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

        avg_train_loss = total_train_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Train Loss: {avg_train_loss:.4f}")

    print(f"\n[INFO] Saving model checkpoints to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(label2id, file, indent=4)

    saved_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"[SUCCESS] Saved files: {saved_files}")
    print("[DONE] BERT training completed successfully.")

if __name__ == "__main__":
    main()

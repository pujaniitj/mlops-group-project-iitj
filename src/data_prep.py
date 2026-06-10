"""
data_prep.py
Downloads the IMDB sentiment dataset from Hugging Face, tokenizes it
using the DistilBERT tokenizer, and saves the prepared dataset locally.

Outputs:
  - data/id2label.json     (commit this to the repo)
  - data/tokenized/        (do NOT commit, large files; listed in .gitignore)

Usage:
  python src/data_prep.py
"""
import json
import os
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = "data"
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized")
MAX_LENGTH = 256  # truncate reviews to 256 tokens to fit Kaggle GPU memory


def inspect_raw_data(dataset):
    """Print basic stats so we know what we're working with."""
    print("=" * 60)
    print("RAW DATASET INSPECTION")
    print("=" * 60)
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples:  {len(dataset['test'])}")
    print(f"  Columns:       {dataset['train'].column_names}")

    train_counts = Counter(dataset['train']['label'])
    test_counts = Counter(dataset['test']['label'])
    print(f"  Train class distribution: {dict(train_counts)}")
    print(f"  Test class distribution:  {dict(test_counts)}")

    sample_lengths = [len(x.split()) for x in dataset['train']['text'][:1000]]
    print(f"  Review word count (first 1000 samples): "
          f"min={min(sample_lengths)}, max={max(sample_lengths)}, "
          f"avg={sum(sample_lengths) / len(sample_lengths):.0f}")
    print()


def save_label_mapping():
    """IMDB has 2 classes: 0=negative, 1=positive. Save the mapping."""
    print("=" * 60)
    print("SAVING LABEL MAPPING")
    print("=" * 60)
    id2label = {"0": "negative", "1": "positive"}
    label2id = {"negative": 0, "positive": 1}

    os.makedirs(DATA_DIR, exist_ok=True)
    mapping_path = os.path.join(DATA_DIR, "id2label.json")
    with open(mapping_path, "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"  Saved -> {mapping_path}")
    print(f"  Contents: {id2label}")
    print()


def split_train_val(dataset, val_fraction=0.1, seed=42):
    """Split the training set into train and validation (90/10)."""
    print("=" * 60)
    print("SPLITTING TRAIN INTO TRAIN/VAL")
    print("=" * 60)
    split = dataset['train'].train_test_split(test_size=val_fraction, seed=seed)
    print(f"  Train: {len(split['train'])} | Val: {len(split['test'])} | "
          f"Test: {len(dataset['test'])}")
    print()
    return split['train'], split['test'], dataset['test']


def tokenize_splits(train_ds, val_ds, test_ds):
    """Tokenize all three splits with the DistilBERT tokenizer."""
    print("=" * 60)
    print(f"TOKENIZING WITH {MODEL_NAME}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
        )

    print("  Tokenizing train split...")
    train_tok = train_ds.map(tokenize_fn, batched=True)
    print("  Tokenizing val split...")
    val_tok = val_ds.map(tokenize_fn, batched=True)
    print("  Tokenizing test split...")
    test_tok = test_ds.map(tokenize_fn, batched=True)
    print()
    return train_tok, val_tok, test_tok


def save_tokenized(train_tok, val_tok, test_tok):
    """Save tokenized datasets to disk (NOT committed to repo)."""
    print("=" * 60)
    print("SAVING TOKENIZED DATASETS")
    print("=" * 60)
    os.makedirs(TOKENIZED_DIR, exist_ok=True)
    train_tok.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
    val_tok.save_to_disk(os.path.join(TOKENIZED_DIR, "val"))
    test_tok.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))
    print(f"  Saved to: {TOKENIZED_DIR}/")
    print(f"  (These are large -- do NOT commit them.)")
    print()


def main():
    print("\n>>> Loading IMDB dataset from Hugging Face Hub...\n")
    dataset = load_dataset("imdb")

    inspect_raw_data(dataset)
    save_label_mapping()
    train_ds, val_ds, test_ds = split_train_val(dataset)
    train_tok, val_tok, test_tok = tokenize_splits(train_ds, val_ds, test_ds)
    save_tokenized(train_tok, val_tok, test_tok)

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("  Commit to repo:  data/id2label.json")
    print("  Do NOT commit:   data/tokenized/  (large files)")


if __name__ == "__main__":
    main()

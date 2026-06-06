"""
train.py
Skeleton for fine-tuning DistilBERT on IMDB.

This is the LOCAL / REFERENCE version. Actual fine-tuning happens in a
Kaggle Notebook (see notebooks/) where the free GPU is available.

This script verifies that the model loads correctly with the right
number of labels from data/id2label.json.

Usage:
  python src/train.py
"""
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"
MAPPING_PATH = "data/id2label.json"


def load_label_mapping(path=MAPPING_PATH):
    """Load id2label and label2id mappings from JSON."""
    with open(path) as f:
        m = json.load(f)
    id2label = {int(k): v for k, v in m['id2label'].items()}
    label2id = m['label2id']
    return id2label, label2id


def load_model_and_tokenizer():
    """Load DistilBERT tokenizer and classification head with correct labels."""
    id2label, label2id = load_label_mapping()
    num_labels = len(id2label)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Number of labels: {num_labels}")
    print(f"id2label: {id2label}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model and tokenizer loaded successfully.")
    print(f"Total parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")
    print(f"\nNext step: copy this code into a Kaggle notebook and add the")
    print(f"Hugging Face Trainer setup (see project doc Task 4).")


if __name__ == "__main__":
    main()

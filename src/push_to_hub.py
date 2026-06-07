import os
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi

HF_REPO = "pujaniitj/MLOPS_GROUP_PROJECT"

MODEL_DIR = "./final_model_v1"


def verify_local_model(model_dir: str) -> None:
    
    if not os.path.isdir(model_dir):
        sys.exit(
            f"\n Model directory '{model_dir}' not found.\n"
            f"   Did you extract Pujan's final_model_v1.zip into '{model_dir}/'?"
        )

    expected_files = ["config.json", "tokenizer_config.json"]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        sys.exit(
            f"\n Missing files in '{model_dir}': {missing}\n"
            f"   Make sure you extracted ALL files from final_model_v1.zip."
        )


def main():
    verify_local_model(MODEL_DIR)

    print(f"Loading model and tokenizer from {MODEL_DIR}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"  Number of labels: {model.config.num_labels}")
    print(f"  id2label: {model.config.id2label}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print(f"\nVerifying authentication with Hugging Face...")
    api = HfApi()
    try:
        user = api.whoami()
        print(f"  Logged in as: {user['name']}")
    except Exception as e:
        sys.exit(
            f"\n Not authenticated with Hugging Face.\n"
            f"   Run: huggingface-cli login\n"
            f"   Error: {e}"
        )

    print(f"\nPushing model to: {HF_REPO}")
    print("(This uploads ~270 MB and may take 1–3 minutes depending on your connection.)")

    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)

    full_url = f"https://huggingface.co/{HF_REPO}"
    print(f"\n Push complete!")
    print(f"   Model URL: {full_url}")
    print(f"\nNext step: edit the model card (README.md) on the model page.")


if __name__ == "__main__":
    main()
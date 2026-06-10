import os
import sys

from transformers import pipeline


DEFAULT_MODEL = "pujaniitj/MLOPS_GROUP_PROJECT"
DEFAULT_TEXT = (
    "This movie was absolutely fantastic, "
    "the acting was superb and the story kept me hooked throughout."
)


def get_input_text() -> str:
    
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return os.environ.get("INPUT_TEXT", DEFAULT_TEXT)


def main():
    model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    hf_token = os.environ.get("HF_TOKEN")  # only used if repo is private
    text = get_input_text()

    print(f"Model: {model_name}")
    print(f"Input: {text}")
    print(f"\nLoading model from Hugging Face Hub...")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        token=hf_token,
    )

    print(f"Running inference...\n")
    result = classifier(text)

    label = result[0]["label"]
    score = result[0]["score"]

    print("=" * 40)
    print(f"  Sentiment:  {label}")
    print(f"  Confidence: {score:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
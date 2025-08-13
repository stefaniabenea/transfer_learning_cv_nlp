import torch
from transformers import pipeline
import argparse
import sys

parser = argparse.ArgumentParser(description="Run sentiment inference with a fine-tuned DistilBERT.")
parser.add_argument("--text", nargs="*", help="One or more input texts. If omitted, reads from stdin.")
args = parser.parse_args()


clf = pipeline(
    task = "text-classification",
    model="distilbert_imdb_best",   
    tokenizer="distilbert_imdb_best",
    truncation=True,
    max_length=256,
    device=0 if torch.cuda.is_available() else -1
)

if args.text:
    texts = args.text
else:
    texts = [line.strip() for line in sys.stdin if line.strip()]

if not texts:
    print("No input text provided. Use --text '...' or pipe lines via stdin.")
    sys.exit(1)

preds = clf(texts, return_all_scores = True)

for i, p in enumerate(preds):
    print(f"\n#{i+1}: {texts[i]}")
    best = max(p, key=lambda x: x["score"])
    print(f"  {best['label']}: {100*best['score']:.2f}%")
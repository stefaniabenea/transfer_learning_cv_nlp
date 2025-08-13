# Transfer Learning – CV (ResNet18) & NLP (DistilBERT)

This repository contains two minimal, production-style transfer learning pipelines:

- **Computer Vision (PyTorch + Torchvision)**: Fine-tune ResNet18 on CIFAR-10 and run top-K image inference.
- **NLP (Hugging Face Transformers)**: Fine-tune DistilBERT for IMDb sentiment classification and run a text-classification pipeline.


## 1. Environment

# install deps
```
pip install -r requirements.txt
```

## 2. Computer Vision – ResNet18 on CIFAR-10

### Training
```
cd cv
python finetune_resnet.py
```
What it does:
- Loads CIFAR-10 (auto-download).
- Uses ImageNet normalization & light augmentation.
- Starts with frozen backbone; after a few epochs, unfreezes layer4 and continues fine-tuning.
- Tracks best validation accuracy and saves resnet18_cifar10_best.pth.

Expected outputs:
- resnet18_cifar10_best.pth in the working directory.
- Console logs with Train/Test loss & accuracy per epoch.

### Inference (Top-K)
```
python infer_resnet.py --image /path/to/your/image.jpg
```
Output (example):

Predictions:
cat (74.12%)
dog (20.31%)
frog (3.56%)

## 3. NLP - DistilBERT on IMDb (Binary Sentiment)

### Training & Evaluation
```
cd ../nlp
python finetune_bert.py
```
What it does:
- Loads the IMDb dataset (Hugging Face datasets).
- Tokenizes with AutoTokenizer (DistilBERT).
- Trains with AdamW, linear warmup/decay scheduler, gradient clipping.
- Saves the best model and tokenizer to distilbert_imdb_best/.
- Prints test loss/accuracy at the end.

### Inference (Best label)
infer_bert.py uses transformers.pipeline with the fine-tuned model to classify new texts.
Make sure that the `distilbert_imdb_best/` folder exists (created by the training step above).

Run examples:

- Single text from CLI
```
python infer_bert.py --text "This movie was surprisingly good and fun!"
```
- Multiple texts from CLI
```
python infer_bert.py --text "Great film!" "Terrible acting and a boring plot."
```
- From a text file (one review per line) using stdin
```
type reviews.txt | python infer_bert.py   # Windows
cat reviews.txt | python infer_bert.py    # Linux/Mac
```
Output (example):
#1: This movie was surprisingly good and fun!
  POSITIVE: 98.34%

#2: Terrible acting and a boring plot.
  NEGATIVE: 96.12%

## 4. Repo structure
```
transfer_learning/
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore               # Files/folders to ignore in Git
│
├── cv/                      # Computer Vision – ResNet18 on CIFAR-10
│   ├── finetune_resnet.py   # Fine-tune ResNet18 on CIFAR-10
│   └── infer_resnet.py      # Top-K inference with trained ResNet18
│
└── nlp/                     # Natural Language Processing – DistilBERT on IMDb
    ├── finetune_bert.py     # Fine-tune DistilBERT for sentiment classification
    └── infer_bert.py        # Inference script (best label prediction)
```
## 5. Notes
- The cv/ folder handles image classification transfer learning.
- The nlp/ folder handles text classification transfer learning.
- Model weights and tokenizer files are saved in: resnet18_cifar10_best.pth (CV) / distilbert_imdb_best/ (NLP)
- These trained models must exist before running inference scripts.


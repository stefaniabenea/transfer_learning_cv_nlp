import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model_name = "distilbert-base-uncased"
num_labels = 2
batch_size = 16
epochs = 3
lr = 2e-5
weight_decay = 0.01
max_length = 256
warmup_ratio = 0.1

raw_ds = load_dataset('imdb')
raw_ds = raw_ds.rename_column("label","labels")
raw_train = raw_ds["train"]
raw_test = raw_ds["test"]

raw_split = raw_train.train_test_split(test_size=0.1, seed=seed)
train_ds  = raw_split["train"]
val_ds    = raw_split["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,        
        max_length=max_length,  
        
    )

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_tok   = val_ds.map(tokenize_fn,   batched=True, remove_columns=["text"])
test_tok  = raw_test.map(tokenize_fn, batched=True, remove_columns=["text"])

cols = ["input_ids", "attention_mask", "labels"]
train_tok.set_format(type="torch", columns=cols)
val_tok.set_format(type="torch",   columns=cols)
test_tok.set_format(type="torch",  columns=cols)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_tok, batch_size=batch_size, shuffle=True,
    collate_fn=collator, num_workers=0, pin_memory=(device.type=="cuda")
)
val_loader = DataLoader(
    val_tok, batch_size=batch_size, shuffle=False,
    collate_fn=collator, num_workers=0, pin_memory=(device.type=="cuda")
)
test_loader = DataLoader(
    test_tok, batch_size=batch_size, shuffle=False,
    collate_fn=collator, num_workers=0, pin_memory=(device.type=="cuda")
)

id2label = {0: "neg", 1: "pos"}
label2id = {"neg": 0, "pos": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)
no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

num_update_steps_per_epoch = len(train_loader)
t_total = num_update_steps_per_epoch * epochs
warmup_steps = int(warmup_ratio * t_total)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=t_total,
)

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        
        outputs = model(**batch)         
        loss = outputs.loss              
        logits = outputs.logits          

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        optimizer.step()
        scheduler.step()                 

        
        running_loss += loss.item() * batch["input_ids"].size(0)
        preds = logits.argmax(dim=-1)    
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        running_loss += loss.item() * batch["input_ids"].size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return running_loss / total, correct / total



best_val_acc = 0.0
save_dir = "distilbert_imdb_best"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss,   val_acc   = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train - loss: {train_loss:.4f} | acc: {train_acc:.4f}")
    print(f"  Val   - loss: {val_loss:.4f}   | acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(save_dir)      
        tokenizer.save_pretrained(save_dir)  
        print(f"New best saved to: {save_dir}")

best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
best_tokenizer = AutoTokenizer.from_pretrained(save_dir)

test_loss, test_acc = evaluate(best_model, test_loader, device)
print(f"[BEST] Test - loss: {test_loss:.4f} | acc: {test_acc:.4f}")

clf = pipeline(
    "text-classification",
    model="distilbert_imdb_best",   
    tokenizer="distilbert_imdb_best",
    device=0 if torch.cuda.is_available() else -1
)

texts = [
    "This movie was surprisingly good and fun!",
    "Terrible acting and a boring plot.",
]
print(clf(texts))
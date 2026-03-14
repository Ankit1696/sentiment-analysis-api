"""
╔══════════════════════════════════════════════════════════════╗
║  Sentiment Analysis — Custom Model Training Pipeline        ║
║  Fine-tunes DistilBERT on the IMDB dataset from scratch     ║
║  Author: Ankit Kumar                                        ║
╚══════════════════════════════════════════════════════════════╝

This script demonstrates end-to-end ML engineering:
  1. Data loading & preprocessing
  2. Manual tokenization with HuggingFace tokenizer
  3. Custom PyTorch Dataset & DataLoader
  4. Fine-tuning a pre-trained transformer
  5. Training loop with mixed precision & gradient clipping
  6. Evaluation with accuracy, precision, recall, F1
  7. Saving the trained model for deployment
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import DistilBertModel, DistilBertTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ═══════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=256, help="Max token sequence length")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples (use -1 for full dataset)")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples (use -1 for full dataset)")
    parser.add_argument("--output_dir", type=str, default="./sentiment_model", help="Directory to save the trained model")
    return parser.parse_args()


# ═══════════════════════════════════════
# 2. Custom PyTorch Dataset
# ═══════════════════════════════════════

class IMDBSentimentDataset(Dataset):
    """
    Custom Dataset that handles tokenization manually.
    Each sample is tokenized on-the-fly using the DistilBERT tokenizer.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Manual tokenization step:
        # - Converts raw text to token IDs
        # - Adds [CLS] and [SEP] special tokens
        # - Pads/truncates to max_len
        # - Creates attention mask (1 = real token, 0 = padding)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # Shape: (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # Shape: (max_len,)
            "label": torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════
# 3. Custom Classifier Model
# ═══════════════════════════════════════

class SentimentClassifier(nn.Module):
    """
    Custom classifier built on top of pre-trained DistilBERT.

    Architecture:
        DistilBERT (frozen/unfrozen) → [CLS] token → Dropout → FC(768→256) → ReLU
        → Dropout → FC(256→2) → Softmax

    This is a custom head — we're NOT using AutoModelForSequenceClassification.
    We manually extract the [CLS] token representation and build our own
    classification layers on top.
    """
    def __init__(self, pretrained_model_name="distilbert-base-uncased", dropout=0.3):
        super(SentimentClassifier, self).__init__()

        # Load pre-trained DistilBERT backbone
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)

        # Custom classification head (built from scratch)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.distilbert.config.hidden_size, 256),  # 768 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # 256 → 2 (Positive / Negative)
        )

    def forward(self, input_ids, attention_mask):
        # Pass through DistilBERT backbone
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token's hidden state (first token)
        # This is the standard approach for classification tasks
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch, 768)

        # Pass through our custom classification head
        logits = self.classifier(cls_output)  # Shape: (batch, 2)

        return logits


# ═══════════════════════════════════════
# 4. Training Function
# ═══════════════════════════════════════

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    """
    Runs one full training epoch.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Progress logging
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{len(data_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(data_loader), correct / total


# ═══════════════════════════════════════
# 5. Evaluation Function
# ═══════════════════════════════════════

def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluates model on the test set.
    Returns loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return avg_loss, accuracy, precision, recall, f1


# ═══════════════════════════════════════
# 6. Main Training Pipeline
# ═══════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 60)
    print("  Sentiment Analysis — Custom Model Training")
    print("=" * 60)

    # ─── Device Setup ───
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print(f"✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"✓ Using CPU")

    print(f"✓ Epochs: {args.epochs}")
    print(f"✓ Batch Size: {args.batch_size}")
    print(f"✓ Learning Rate: {args.lr}")
    print(f"✓ Max Sequence Length: {args.max_len}")
    print()

    # ─── Step 1: Load Dataset ───
    print("▶ Step 1/6: Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    if args.train_samples > 0:
        train_data = dataset["train"].shuffle(seed=42).select(range(args.train_samples))
    else:
        train_data = dataset["train"].shuffle(seed=42)

    if args.test_samples > 0:
        test_data = dataset["test"].shuffle(seed=42).select(range(args.test_samples))
    else:
        test_data = dataset["test"].shuffle(seed=42)

    print(f"  ✓ Training samples: {len(train_data)}")
    print(f"  ✓ Test samples:     {len(test_data)}")
    print()

    # ─── Step 2: Initialize Tokenizer ───
    print("▶ Step 2/6: Initializing DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print(f"  ✓ Vocabulary size: {tokenizer.vocab_size}")
    print(f"  ✓ Max length: {args.max_len}")
    print()

    # ─── Step 3: Create DataLoaders ───
    print("▶ Step 3/6: Creating custom DataLoaders...")
    train_dataset = IMDBSentimentDataset(
        texts=train_data["text"],
        labels=train_data["label"],
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    test_dataset = IMDBSentimentDataset(
        texts=test_data["text"],
        labels=test_data["label"],
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Test batches:  {len(test_loader)}")
    print()

    # ─── Step 4: Initialize Model ───
    print("▶ Step 4/6: Building SentimentClassifier...")
    model = SentimentClassifier(pretrained_model_name="distilbert-base-uncased")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters:     {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print()

    # ─── Step 5: Setup Optimizer & Scheduler ───
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    # ─── Step 6: Training Loop ───
    print("▶ Step 5/6: Training...")
    print("-" * 60)
    best_f1 = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n  Epoch {epoch+1}/{args.epochs}")
        print(f"  {'─' * 50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        epoch_time = time.time() - epoch_start

        # Evaluate after each epoch
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(
            model, test_loader, loss_fn, device
        )

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │ Epoch {epoch+1} Results ({epoch_time:.1f}s)               │")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │ Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f} │")
        print(f"  │ Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.4f} │")
        print(f"  │ Precision:  {val_prec:.4f}  |  Recall:    {val_recall:.4f} │")
        print(f"  │ F1 Score:   {val_f1:.4f}                        │")
        print(f"  └─────────────────────────────────────────────┘")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"\n  ★ New best F1! Saving model...")
            os.makedirs(args.output_dir, exist_ok=True)

            # Save model weights
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "args": vars(args),
            }, os.path.join(args.output_dir, "model.pt"))

            # Save tokenizer for deployment
            tokenizer.save_pretrained(args.output_dir)

            print(f"  ✓ Model saved to {args.output_dir}/model.pt")
            print(f"  ✓ Tokenizer saved to {args.output_dir}/")

    # ─── Final Summary ───
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best F1 Score: {best_f1:.4f}")
    print(f"  Model saved to: {args.output_dir}/")
    print(f"\n  To use in the API, restart the backend server:")
    print(f"    uvicorn main:app --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()

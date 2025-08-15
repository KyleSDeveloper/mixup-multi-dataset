import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from rich.console import Console

from src.data import get_loaders
from src.models import create_model
from src.mixup import mixup_batch
from src.metrics import accuracy, ece
from src.utils.seed import set_seed
from src.utils.io import save_checkpoint, append_row_csv

console = Console()

def train_one_epoch(model, loader, optimizer, device, mixup_alpha=0.0, label_smoothing=0.0):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for x, y in loader:
        if isinstance(y, (tuple, list)):  # Oxford-IIIT Pet returns (category, ...)
            y = y[0]
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        if mixup_alpha > 0.0:
            x, y_a, y_b, _, lam = mixup_batch(x, y, alpha=mixup_alpha)
            logits = model(x)
            loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)
            preds = logits.argmax(dim=1)
            acc = lam * (preds == y_a).float().mean().item() + (1 - lam) * (preds == y_b).float().mean().item()
        else:
            logits = model(x)
            loss = ce(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += acc * bs
        total += bs

    return running_loss / total, running_acc / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, loss_sum, acc_sum = 0, 0.0, 0.0
    logits_list, targets_list = [], []
    for x, y in loader:
        if isinstance(y, (tuple, list)):
            y = y[0]
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='mean')
        acc = accuracy(logits, y)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum += acc * bs
        total += bs
        logits_list.append(logits.cpu())
        targets_list.append(y.cpu())
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    e = ece(logits, targets)
    return loss_sum / total, acc_sum / total, e

def run_one_seed(args, seed: int):
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, weights = create_model(num_classes=1, model_name=args.model_name, pretrained=args.pretrained)

    # Build loaders once to get num_classes
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers, args.image_size, weights, seed=seed
    )
    # Recreate model with correct head
    model, weights = create_model(num_classes=num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_epoch = -1
    patience = args.patience
    last_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing
        )
        val_loss, val_acc, val_ece = evaluate(model, val_loader, device)
        scheduler.step()

        console.print(f"[epoch {epoch+1:03d}] "
                      f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} acc={val_acc:.4f} ece={val_ece:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            last_improve = 0
            ckpt_path = Path(args.checkpoint_dir) / f"{args.dataset}_seed{seed}_alpha{args.mixup_alpha}.pt"
            save_checkpoint(model, ckpt_path)
        else:
            last_improve += 1

        if last_improve >= patience:
            console.print(f"[bold yellow]Early stopping at epoch {epoch+1} (best @ {best_epoch+1})[/bold yellow]")
            break

    # Load best and evaluate test
    best_ckpt = Path(args.checkpoint_dir) / f"{args.dataset}_seed{seed}_alpha{args.mixup_alpha}.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
    model.to(device)
    test_loss, test_acc, test_ece = evaluate(model, test_loader, device)

    # Record results
    row = {
        "dataset": args.dataset,
        "seed": seed,
        "mixup_alpha": args.mixup_alpha,
        "epochs": args.epochs,
        "best_epoch": best_epoch + 1,
        "val_best_loss": round(best_val_loss, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "ece_test": round(test_ece, 6),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "image_size": args.image_size,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "timestamp": int(time.time())
    }
    append_row_csv(args.results_file, row)
    console.print(f"[bold green]Saved results to {args.results_file}[/bold green]")
    console.print(row)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['cifar10','svhn','pets'])
    ap.add_argument('--data-root', type=str, default='data')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--mixup-alpha', type=float, default=0.0)
    ap.add_argument('--label-smoothing', type=float, default=0.0)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=160)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--model-name', type=str, default='resnet18')
    ap.add_argument('--pretrained', action='store_true', default=True)
    ap.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    ap.add_argument('--results-file', type=str, default='results/results.csv')
    ap.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        run_one_seed(args, seed)

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from src.data import get_loaders
from src.models import create_model
from src.metrics import accuracy, ece

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['cifar10','svhn','pets'])
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--data-root', type=str, default='data')
    ap.add_argument('--image-size', type=int, default=160)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--model-name', type=str, default='resnet18')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, weights = create_model(num_classes=1, model_name=args.model_name, pretrained=True)  # will reset head below
    # infer num_classes by building loaders once
    _, _, test_loader, num_classes = get_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers, args.image_size, weights)
    model, weights = create_model(num_classes=num_classes, model_name=args.model_name, pretrained=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.to(device)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            if isinstance(y, tuple) or isinstance(y, list):
                y = y[0]  # Oxford-IIIT Pet returns (category, ...)
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    loss = F.cross_entropy(logits, targets).item()
    acc = accuracy(logits, targets)
    e = ece(logits, targets)

    print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.4f} | ECE: {e:.4f}")

if __name__ == "__main__":
    main()

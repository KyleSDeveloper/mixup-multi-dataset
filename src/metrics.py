import torch
import torch.nn.functional as F

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def ece(logits, targets, n_bins: int = 15):
    # Expected Calibration Error with equal-width bins on confidence
    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    targets = targets.view(-1)
    ece_val = 0.0
    bins = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if mask.any():
            acc = (preds[mask] == targets[mask]).float().mean()
            conf = confs[mask].mean()
            ece_val += (mask.float().mean() * (conf - acc).abs()).item()
    return float(ece_val)

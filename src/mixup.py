import torch

def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, None, None, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, idx, lam

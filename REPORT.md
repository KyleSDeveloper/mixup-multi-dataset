# Mixup Across Multiple Image Datasets — Report

## Datasets
- CIFAR-10
- SVHN
- Oxford-IIIT Pet

## Setup
- Backbone: ResNet-18 (pretrained=True)
- Epochs: 30
- Image size: 160 (224 for Pet)
- Seeds: 0, 1, 2
- Mixup alpha: 0.2

## Results (mean ± std over seeds)

| Dataset | Baseline Acc | Mixup Acc | Δ Acc | Baseline LogLoss | Mixup LogLoss | Δ LogLoss | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 |  |  |  |  |  |  |  |
| SVHN     |  |  |  |  |  |  |  |
| Pets     |  |  |  |  |  |  |  |

## Takeaways
- …

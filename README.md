# Mixup Across Multiple Image Datasets (Mini-Benchmark)

This repo is a small, reproducible study testing whether **mixup** improves validation loss, accuracy, and calibration across multiple image classification datasets (e.g., CIFAR-10, SVHN, Oxford-IIIT Pet).

## TL;DR
- One command per run (baseline vs. mixup).
- Same backbone and training budget across datasets for a fair comparison.
- Results append to `results/results.csv` for easy notebook analysis.

## Quickstart

### 1) Create env (Conda)
```bash
conda create -n mixup-bench python=3.12 -y
conda activate mixup-bench
# Install PyTorch + TorchVision as recommended for your system:
# See: https://pytorch.org/get-started/locally/
# Example (CPU-only):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then the rest:
pip install -r requirements.txt
```

### Or with uv
```bash
uv venv .venv -p 3.12
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

### 2) Train (baseline vs. mixup)
```bash
# Baseline (no mixup)
python -m src.train --dataset cifar10 --epochs 30 --batch-size 128 --seeds 0 1 2

# Mixup (alpha=0.2)
python -m src.train --dataset cifar10 --epochs 30 --batch-size 128 --seeds 0 1 2 --mixup-alpha 0.2
```

Other datasets:
```bash
# SVHN
python -m src.train --dataset svhn --epochs 30 --batch-size 128 --seeds 0 1 2
python -m src.train --dataset svhn --epochs 30 --batch-size 128 --seeds 0 1 2 --mixup-alpha 0.2

# Oxford-IIIT Pet
python -m src.train --dataset pets --epochs 30 --batch-size 64 --seeds 0 1 2 --image-size 224
python -m src.train --dataset pets --epochs 30 --batch-size 64 --seeds 0 1 2 --image-size 224 --mixup-alpha 0.2
```

### 3) Aggregate & visualize
Run your experiments; all results append to `results/results.csv`. Then open the analysis notebook:
```bash
# (Optional) Generate a quick report table
python -m src.utils.aggregate --results-file results/results.csv

# Explore in notebook
jupyter lab notebooks/01_explore_results.ipynb
```

## Project Goals
- **Compare** baseline vs. mixup under matched compute across 3 datasets.
- **Report** mean ± std over seeds for Acc, LogLoss, and ECE (calibration).
- **Communicate** concise findings in your README/report.

## Repo Layout
```
mixup-multi-dataset/
  ├─ src/
  │   ├─ train.py          # training loop w/ optional mixup
  │   ├─ evaluate.py       # load checkpoint & evaluate
  │   ├─ data.py           # dataset/transform registry
  │   ├─ models.py         # model factory (resnet18)
  │   ├─ mixup.py          # mixup utilities
  │   ├─ metrics.py        # accuracy, ECE, helpers
  │   └─ utils/
  │       ├─ seed.py       # deterministic seeding
  │       ├─ io.py         # CSV append, checkpoint utils
  │       └─ aggregate.py  # summarize results.csv
  ├─ notebooks/
  │   └─ 01_explore_results.ipynb
  ├─ scripts/
  │   └─ run_grid.sh       # example grid for 3 datasets, baseline/mixup
  ├─ results/              # CSV results
  ├─ checkpoints/          # best model per run
  ├─ data/                 # datasets cache (auto-downloaded by torchvision)
  ├─ requirements.txt
  ├─ environment.yml
  ├─ LICENSE
  └─ README.md
```

## Notes
- Default image size is 160 for speed on CPU; use `--image-size 224` for best transfer.
- Early stopping by val loss is enabled with `--patience` (default 5).
- For a tiny hyperparam sweep of `alpha`, try `scripts/run_grid.sh` or Optuna later.

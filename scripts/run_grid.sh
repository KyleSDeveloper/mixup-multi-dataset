#!/usr/bin/env bash
set -e

# Baseline and mixup runs across 3 datasets
DATASETS=("cifar10" "svhn" "pets")
SEEDS=("0" "1" "2")

for DS in "${DATASETS[@]}"; do
  for ALPHA in "0.0" "0.2"; do
    for SEED in "${SEEDS[@]}"; do
      echo "Running dataset=${DS}, alpha=${ALPHA}, seed=${SEED}"
      python -m src.train --dataset ${DS} --mixup-alpha ${ALPHA} --seeds ${SEED}
    done
  done
done

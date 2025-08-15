import os
import csv
from pathlib import Path
from typing import Dict, Any
import torch

def ensure_parent(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_checkpoint(model, path: str | Path):
    p = ensure_parent(path)
    torch.save(model.state_dict(), p)

def append_row_csv(path: str | Path, row: Dict[str, Any]):
    p = ensure_parent(path)
    write_header = not p.exists()
    with p.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

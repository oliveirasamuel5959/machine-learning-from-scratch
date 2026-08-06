import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from mlscratch.deep_learning.dataset_download import build_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==== Params =====
batch_size = 16

# ====== Load Dataset ========
print(f"[INFO] Building Dataset to path {DATA_DIR}")
train_dataset, test_dataset = build_dataset(data_dir=DATA_DIR)
print(f"num samples: {len(train_dataset)}")

# ====== Create Dataloader with batch size =========
print(f"[INFO] Building Dataloader with batch size of {batch_size}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get the first batch to inspect
images, labels = next(iter(train_loader))
print(f"[INFO] Image and label shape: {images.shape}, {labels.shape}")
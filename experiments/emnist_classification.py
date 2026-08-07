import os
import argparse
import time
import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mlscratch.deep_learning.dataset_download import build_dataset
from mlscratch.common.logger import get_logger
from mlscratch.deep_learning.models.mlp import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

def main():
  parser = _get_parser()
  args = parser.parse_args()
  LOGGER = get_logger(__name__)
  
  LOGGER.debug(os.environ)
  LOGGER.debug(args)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.bfloat16
  
  torch.manual_seed(args.seed)
  
  # Note: Initializing an **untrained** model
  model = build_model()
  model = model.to(device)
  
  LOGGER.info(f"Training {sum(p.numel() for p in model.parameters())} model parameters")
  
  model = torch.compile(model)
  
  LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")
  
  train_dataset, test_dataset = build_dataset(data_dir=DATA_DIR)
  LOGGER.debug(f"{len(train_dataset)} training samples")
  LOGGER.debug(f"{len(test_dataset)} training samples")
  
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
  
  LOGGER.info(f"{len(train_loader)} batches per epoch")
  LOGGER.info(f"{len(test_loader)} batches per epoch")
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
  
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=args.lr * 1e-2
  )
  
  is_experiment = False
  exp_dir: Path = Path(args.save_dir)
  if args.experiment_name is not None:
    is_experiment = True
    exp_dir = exp_dir / args.experiment_name
    
def get_mem_stats(device):
  if device.type != "cuda":
    return {"curr_alloc_gb": 0.0}

  props = torch.cuda.get_device_properties(device)

  return {
    "curr_alloc_gb": torch.cuda.memory_allocated(device) / 1024**3,
    "total_gb": props.total_memory / 1024**3,
  }

def _get_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--experiment-name", default=None)
  parser.add_argument("-d", "--dataset-name", default=None)
  parser.add_argument("-m", "--model-name", default=None)
  parser.add_argument("--save-dir", default="../outputs", required=True)
  parser.add_argument("--seed", default=42, type=int)
  parser.add_argument("--num-epochs", default=10, type=int)
  parser.add_argument("--lr", default=3e-5, type=float)
  parser.add_argument("-b", "--batch-size", default=1, type=int)
  return parser
  
if __name__ == "__main__":
  main()
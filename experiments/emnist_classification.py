import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mlscratch.deep_learning.datasets.dataset_download import build_dataset
from mlscratch.common.logger import get_logger
from mlscratch.deep_learning.models.mlp import MLP
from mlscratch.deep_learning.train import train

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
  model = MLP(num_classes=62)
  model = model.to(device)
  
  LOGGER.info(f"Training {sum(p.numel() for p in model.parameters())} model parameters")
  
  # model = torch.compile(model)
  
  LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")
  
  train_dataset, test_dataset = build_dataset(data_dir=DATA_DIR)
  LOGGER.info(f"{len(train_dataset)} training samples")
  LOGGER.info(f"{len(test_dataset)} test samples")
  
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
  
  image, label = next(iter(train_loader)) 
                      
  LOGGER.info(f"{len(train_loader)} batches per epoch with image and label shape {image.shape}, {label.shape}")
  LOGGER.info(f"{len(test_loader)} batches per epoch with image and label shape {image.shape}, {label.shape}")
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
  loss_fn = nn.CrossEntropyLoss()
  
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=args.lr * 1e-2
  )
  
  is_experiment = False
  exp_dir: Path = Path(args.save_dir)
  if args.experiment_name is not None:
    is_experiment = True
    exp_dir = exp_dir / args.experiment_name
    
  # =========
  # Train 
  # =========
  for epoch in tqdm(range(args.num_epochs)):
    LOGGER.info(f"Begin epoch {epoch}")
    
    train_loss, train_acc = train(
      model=model,
      dataloader=train_loader,
      loss_fn=loss_fn,
      opitmizer=optimizer,
      device=device
    )
    
    # Print out training
    print(
      f"Epoch: {epoch+1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
    )
  
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
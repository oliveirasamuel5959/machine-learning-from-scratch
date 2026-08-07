"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
# from torch.utils.tensorboard import SummaryWriter

def eval(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  device: torch.device) -> Tuple[float, float]:
  
  model.eval()
  
  total_loss = 0.0
  correct = 0
  total = 0 
  
  with torch.no_grad():
    # Loop through DataLoader batches
    for batch, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False):
      images, labels = images.to(device), labels.to(device)
      
      # 1. Forward pass
      logits = model(images)
      
      # 2. Calculate and accumulate loss
      loss = loss_fn(logits, labels)
      batch_size = labels.size(0)
      total_loss += loss.item() * batch_size
      
      # Calculate and accumulate accuracy
      preds = logits.argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += batch_size
    
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = total_loss / total
    test_acc = correct / total
    
    return test_loss, test_acc
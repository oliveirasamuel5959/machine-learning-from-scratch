import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional


def train(model, dataloader, loss_fn, opitmizer, device):
  running_loss = 0
  total = 0
  
  model.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  
  for batch_idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False):
    images, labels = images.to(device), labels.to(device)
    
    # 1. Forward pass
    y_pred = model(images)
    
    # 2. Calculate and accumulate loss
    loss = loss_fn(y_pred, labels)
    train_loss += loss.item()
    
    # 3. Optimizer zero grad
    opitmizer.zero_grad()
    
    # 4. Loss backward
    loss.backward()
    
    # 5. opitimizer step
    opitmizer.step()
    
    # Calculate and accumulate accuracy metric accros all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == labels).sum().item() / len(y_pred)
    
    
  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  
  return train_loss, train_acc
    
    
    
    
    
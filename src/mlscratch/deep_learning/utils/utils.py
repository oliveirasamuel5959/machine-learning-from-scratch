import os
import torch
import json

def load_to_device(path, device):
  return torch.load(path / "ckpts" / "model.pt", map_location=device, weights_only=True)

def save_model(model, path):
  path = path / "ckpts"
  os.makedirs(path, exist_ok=True)
  torch.save(model.state_dict(), path / "model.pt")

def save_metrics(path, metrics_results, prefix):
  with open(path / f"{prefix}.json", "w") as fp:
    json.dump(metrics_results, fp, indent=4)
    
def print_train_time(start, end, device=None):
  """Prints difference between start and end time.

  Args:
      start (float): Start time of computation (preferred in timeit format). 
      end (float): End time of computation.
      device ([type], optional): Device that compute is running on. Defaults to None.

  Returns:
      float: time between start and end in seconds (higher is longer).
  """
  total_time = end - start
  print(f"\nTrain time on {device}: {total_time:.3f} seconds")
  return total_time
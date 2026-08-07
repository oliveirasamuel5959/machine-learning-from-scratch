import os
import torch
import json

def load_to_device(model, path, device, args):
  return torch.load(path, map_location=device, weights_only=True)

def save_model(model, path):
  path = path / "ckpts"
  os.makedirs(path, exist_ok=True)
  torch.save(model.state_dict(), path / "model.pt")

def save_metrics(path, metrics_results):
  with open(path / "metrics.json", "w") as fp:
    json.dump(metrics_results, fp, indent=4)
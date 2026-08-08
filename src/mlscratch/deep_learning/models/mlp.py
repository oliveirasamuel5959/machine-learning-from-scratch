import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim=784, num_classes=10):
    super(MLP, self).__init__()
    self.flatten = nn.Flatten()
    self.input = nn.Linear(input_dim, 256)
    self.relu = nn.ReLU()
    self.hidden = nn.Linear(256, 128)
    self.relu = nn.ReLU()
    self.output = nn.Linear(128, num_classes)
    
  def forward(self, x):
    x = self.flatten(x) # Flatten (N, 1, 28, 28) -> (N, 784)
    x = self.input(x)
    x = self.relu(x)
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x
  
def main():
  model = MLP(input_dim=784, num_classes=62)
  
  print("\n ====== [MODEL] Architecture =======")
  print(model)
  
  print("\n ====== [MODEL] Layers details =======")
  for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
  
  total_params = sum(param.numel() for param in model.parameters()) / 1e6
  print(f"\n[MODEL] Total parameters: {total_params:.2f}M")
  
  print(f"\n[MODEL] Size in memory assuming 32-bit float precision: {total_params}")
  model_size_mb = (total_params * 4) / (1024 ** 2)
  print(f"Model size: {model_size_mb:.2f} MB")
  
if __name__ == '__main__':
  main()
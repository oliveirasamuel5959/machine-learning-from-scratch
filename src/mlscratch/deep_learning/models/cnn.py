import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    
  def forward(self, x):
    x = self.block(x)
    return x
  
class CNNClassifier(nn.Module):
  def __init__(self, in_channels=3, num_classes=10) -> None:
    super(CNNClassifier, self).__init__()
    
    self.features = nn.Sequential(
      ConvBlock(in_channels=in_channels, out_channels=32),
      ConvBlock(in_channels=32, out_channels=64),
      ConvBlock(in_channels=64, out_channels=128)
    )
    
    self.classifier = nn.Sequential(
      nn.Linear(128 * 8 * 8, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, num_classes)
    )
    
  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

def main():
  model = CNNClassifier(in_channels=3, num_classes=62)
  
  print("\n ====== [MODEL] Architecture =======")
  print(model)
  
  print("\n ====== [MODEL] Layers details =======")
  for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
  
  total_params = sum(param.numel() for param in model.parameters())
  total_params_M = total_params / 1e6
  print(f"\n[MODEL] Total parameters: {total_params_M:.2f}M")
  
  print(f"\n[MODEL] Size in memory assuming 32-bit float precision")
  model_size_mb = (total_params * 4) / (1024 ** 2)
  print(f"Model size: {model_size_mb:.2f} MB")
  
if __name__ == '__main__':
  main()
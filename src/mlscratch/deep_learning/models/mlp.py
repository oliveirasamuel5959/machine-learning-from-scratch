import torch
import torch.nn as nn

class MLP(nn.Module):
  def __int__(self, input_dim=784, num_classes=10):
    super(MLP, self).__init__()
    self.flatten = nn.Flatten()
    self.input = nn.Linear(input_dim, 256)
    self.relu = nn.ReLU()
    self.hidden = nn.Linear(256, 128)
    self.relu = nn.ReLU()
    self.output = nn.Linear(128, num_classes)
    
  def forward(self, x):
    x = self.input(x)
    x = self.relu(x)
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x
  
def build_model():
  model = MLP(num_classes=62)
  return model
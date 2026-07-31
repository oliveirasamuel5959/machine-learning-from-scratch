import torch
import torch.nn as nn

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.hidden = nn.Linear(2, 4) # Input to hidden layer  
    self.relu = nn.ReLU()
    self.output = nn.Linear(4, 1) # Hidden to output layer
    
  def forward(self, x):
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return x
  
def main():
  model = SimpleNN()
  
  # Shape: (batch_size, in_features) -> (4, 2)
  # input_data = torch.tensor([[4.0, 5.2]])
  input_data = torch.randn(32, 2)
  
  # 3. Pass data through the linear layer
  output_data = model(input_data)
  
  # Shape: (batch_size, out_features) -> (4, 2)
  print("Output shape:", output_data.shape)
  print("Output", output_data[0].item())
  
if __name__ == '__main__':
  main()
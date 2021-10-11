from torch import nn
import torch

class Identity(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x

class ExpandDim(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return torch.unsqueeze(x, dim=self.dim)
    

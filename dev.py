import torch


x = torch.arange(0, 15, 5, dtype=torch.float32)
print(x.repeat(32, 1))
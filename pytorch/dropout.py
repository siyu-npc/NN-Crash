import torch
import torch.nn as nn

dropout = nn.Dropout(0.5, inplace=False)
input = torch.randn(2, 2, 2, 2)
print(input)
output = dropout(input)
print(output)
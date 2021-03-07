import torch.nn as nn
import torch

input = torch.Tensor([1., 2., 3.,
                      4., 5., 6.,
                      7., 8., 9.])
unpool = nn.MaxUnpool2d(2, 2)
output = unpool(input)

print(output)
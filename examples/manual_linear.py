import torch
from torch import nn
torch.manual_seed(0)
layer=nn.Linear(8,4)
x=torch.randn(3,8)
with torch.no_grad():
    manual=x@layer.weight.T+layer.bias
    builtin=layer(x)
torch.testing.assert_close(manual,builtin)
print("PASS",manual.shape)
# Exercise: inspect the profiler and identify the underlying matrix-multiply/add operations.

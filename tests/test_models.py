import torch
from torch import nn

def test_linear_shape_and_backward():
    torch.manual_seed(1)
    m=nn.Linear(8,3)
    x=torch.randn(5,8,requires_grad=True)
    y=m(x)
    assert y.shape==(5,3)
    y.square().mean().backward()
    assert x.grad is not None

def test_cpu_cuda_agree_if_available():
    if not torch.cuda.is_available(): return
    torch.manual_seed(2)
    m=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2)).eval()
    x=torch.randn(7,4)
    cpu=m(x)
    gpu=m.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(cpu,gpu,rtol=1e-4,atol=1e-5)

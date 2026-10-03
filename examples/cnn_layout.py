import torch
from torch import nn
device="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)).to(device).eval()
x=torch.randn(16,3,224,224,device=device)
with torch.inference_mode():
    nchw=model(x)
    x_cl=x.to(memory_format=torch.channels_last)
    model_cl=model.to(memory_format=torch.channels_last)
    nhwc_like=model_cl(x_cl)
torch.testing.assert_close(nchw,nhwc_like,rtol=1e-4,atol=1e-4)
print("NCHW stride:",x.stride())
print("channels_last stride:",x_cl.stride())
print("PASS; benchmark/profile both layouts on your hardware.")

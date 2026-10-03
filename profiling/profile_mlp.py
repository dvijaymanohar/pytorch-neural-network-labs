import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

device="cuda" if torch.cuda.is_available() else "cpu"
acts=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device=="cuda" else [])
m=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,1024)).to(device).eval()
x=torch.randn(256,1024,device=device)
for _ in range(5):m(x)
with profile(activities=acts,record_shapes=True,profile_memory=True) as p:
    with record_function("mlp_inference"):
        with torch.inference_mode():
            for _ in range(10):m(x)
if device=="cuda":torch.cuda.synchronize()
print(p.key_averages().table(sort_by="self_cuda_time_total" if device=="cuda" else "self_cpu_time_total",row_limit=15))

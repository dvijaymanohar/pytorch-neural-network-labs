import statistics, time, torch
from torch import nn

device="cuda" if torch.cuda.is_available() else "cpu"
model=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,1024)).to(device).eval()
x=torch.randn(256,1024,device=device)

@torch.inference_mode()
def run():
    return model(x)

for _ in range(10): run()
if device=="cuda": torch.cuda.synchronize()
samples=[]
for _ in range(30):
    t0=time.perf_counter(); run()
    if device=="cuda": torch.cuda.synchronize()
    samples.append((time.perf_counter()-t0)*1000)
print({"device":device,"median_ms":statistics.median(samples),"min_ms":min(samples),"max_ms":max(samples)})

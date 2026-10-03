import time, torch
from torch import nn

device="cuda" if torch.cuda.is_available() else "cpu"
model=nn.Sequential(nn.Linear(2048,4096),nn.GELU(),nn.Linear(4096,2048)).to(device).eval()
x=torch.randn(128,2048,device=device)

@torch.inference_mode()
def run(fp16=False):
    if device=="cuda":
        with torch.autocast("cuda",dtype=torch.float16,enabled=fp16):
            return model(x)
    return model(x)

ref=run(False)
test=run(True) if device=="cuda" else ref
torch.testing.assert_close(test.float(),ref.float(),rtol=2e-2,atol=2e-2)
for mode in ([False,True] if device=="cuda" else [False]):
    for _ in range(10):run(mode)
    if device=="cuda":torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(50):run(mode)
    if device=="cuda":torch.cuda.synchronize()
    print({"device":device,"autocast_fp16":mode,"avg_ms":(time.perf_counter()-t0)*1000/50})

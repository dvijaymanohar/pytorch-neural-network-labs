import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
y = x.to(device)
z = (y * 2 + 1).sum()
print("device:", device)
print("shape:", tuple(y.shape), "stride:", y.stride(), "contiguous:", y.is_contiguous())
print("result:", z.item())
if device == "cuda":
    print("allocated_bytes:", torch.cuda.memory_allocated())

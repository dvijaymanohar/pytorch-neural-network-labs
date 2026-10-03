import torch
torch.manual_seed(0)
x=torch.tensor([2.0,-1.0,3.0],requires_grad=True)
w=torch.tensor([0.5,2.0,-1.0],requires_grad=True)
y=(x*w).sum()
loss=(y-1.0).square()
loss.backward()
print("y:",y.item(),"loss:",loss.item())
print("dL/dx:",x.grad)
print("dL/dw:",w.grad)
# Exercise: derive both gradients by hand and verify the chain rule.

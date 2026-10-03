import argparse
import torch
from torch import nn

def make_data(n=1024):
    torch.manual_seed(0)
    x = torch.randn(n, 16)
    w = torch.randn(16, 4)
    y = x @ w + 0.05 * torch.randn(n, 4)
    return x, y

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=100)
    args=p.parse_args()
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    x,y=make_data(); x=x.to(device); y=y.to(device)
    model=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,4)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-2)
    loss_fn=nn.MSELoss()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        pred=model(x); loss=loss_fn(pred,y)
        loss.backward(); opt.step()
    print(f"device={device} final_loss={loss.item():.6f}")
if __name__=="__main__": main()

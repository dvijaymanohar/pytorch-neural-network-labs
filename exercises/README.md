# Exercises

- Inspect tensor shape/stride/contiguity and predict when copies occur.
- Derive gradients for a scalar example, then verify with autograd.
- Implement a Linear layer with explicit matrix multiply + bias and compare.
- Add mixed precision with `torch.autocast`; validate loss/output tolerance.
- Profile an MLP and explain CPU vs GPU time.
- Build a small CNN, compare NCHW vs channels-last where supported, and profile.
- Advanced: connect one custom CUDA/Triton operation and verify against PyTorch.

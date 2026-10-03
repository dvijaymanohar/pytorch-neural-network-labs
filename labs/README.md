# Practical PyTorch labs

Run:
```bash
python examples/autograd_walkthrough.py
python examples/manual_linear.py
python examples/mixed_precision.py
python examples/cnn_layout.py
python profiling/profile_mlp.py
```

Recommended progression:
1. predict shapes/strides and gradients;
2. validate against trusted PyTorch operations;
3. benchmark CPU vs CUDA where available;
4. use PyTorch Profiler to identify operator-level cost;
5. connect expensive operators to CUDA kernels with Nsight when needed.

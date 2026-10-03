# PyTorch Neural Network Labs

Learn how tensors, autograd, neural-network modules, mixed precision, and computer-vision workloads map onto CPU/GPU execution.

## Sequence
tensors/device movement → autograd → linear layer → MLP → training loop → mixed precision → PyTorch Profiler → CNN → image batching/layout → custom extension boundary.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/tensor_basics.py
python examples/mlp_train.py --device auto
pytest -q
```

GPU behavior is measured only when CUDA is available; CPU execution remains a correctness/reference path.

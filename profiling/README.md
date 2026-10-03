# PyTorch profiling

Use `torch.profiler` to connect Python/module activity to CPU and CUDA operators. Begin with end-to-end latency, then inspect operator self time, shapes, memory, and CUDA kernels.

Questions to answer:
- Is time spent in Python/CPU launch overhead or GPU kernels?
- Are unexpected device transfers or synchronizations present?
- Does batching improve utilization?
- Does mixed precision change kernel choice and memory use?

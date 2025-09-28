"""Training hyperparameters and performance tuning flags.

Defaults adjusted for a high-end GPU (e.g., NVIDIA A40). For smaller GPUs:
	- Reduce batch_size (e.g., 256 -> 128 -> 64)
	- Optionally disable mixed precision.
"""

batch_size = 256  # was 32; A40 can handle large batches easily for MNIST-sized data
epochs = 50       # reduce for faster iteration while prototyping
num_workers = 8   # increase data loading parallelism

# Performance flags
pin_memory = True
prefetch_factor = 4
persistent_workers = True
use_amp = True  # automatic mixed precision
grad_accum_steps = 1  # set >1 to simulate larger batch without memory increase 
# SCALE_PLAN

## Decision tree: DDP vs FSDP vs TP/PP

### 1) Does the full model + optimizer state fit on a single GPU at target batch size?
- Yes -> **DDP** (Data Parallel) is the default.
  - Replicate model on each GPU.
  - Scale batch via world_size and/or gradient accumulation.
- No -> go to #2.

### 2) Does it fit if parameters/gradients/optimizer states are sharded across GPUs?
- Yes -> **FSDP** (Fully Sharded Data Parallel).
  - Best when memory (params/optimizer state) is the main bottleneck.
- No -> go to #3.

### 3) Is a single layer too large for one GPU (very large matrices/attention blocks)?
- Consider **Tensor Parallel (TP)**.
  - Split large matrix ops across GPUs.

### 4) Is the model extremely deep and layer partitioning is natural?
- Consider **Pipeline Parallel (PP)**.
  - Split layers across GPUs.
  - Needs microbatching; watch pipeline bubbles.

## What I’d choose for this toy model
- The model fits easily -> **DDP + gradient accumulation** is sufficient.

## Scaling evidence: what I’d measure and why

### Correctness
- **Loss curve consistency** when changing world_size/accum_steps but keeping effective_batch_size constant.
  - Goal: similar training dynamics (within noise).

### Performance
- **Throughput (samples/sec)**: should increase with world_size (up to comm limits).
- **Step time (sec/optimizer step)**: detects all-reduce/communication overhead.
- **Peak memory**: ensures the run is feasible at target batch size.

### Communication overhead (DDP-specific)
- Measure time spent in backward/all-reduce (profiling) to see if comm becomes the bottleneck.

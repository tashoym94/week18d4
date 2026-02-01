# REPRO

## Environment
- OS: macOS
- Python: 3.12 (venv: .venv)
- torch: 2.2.2
- numpy: 1.26.4

## Seeds
- Base seed: 42
- Distributed: seed = 42 + rank

## Commands

### 1) Activate environment
source .venv/bin/activate

### 2) Single-process run
python train.py

### 3) DDP-ready run (world_size=1, local)
torchrun --nproc_per_node=1 train.py

## Outputs
- Writes/overwrites metrics.csv with columns:
  step, loss, effective_batch_size, world_size, accum_steps

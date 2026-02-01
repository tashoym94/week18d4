import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


SEED = 42
EPOCHS = 2
BATCH_SIZE = 32         # micro-batch per process
ACCUM_STEPS = 4         # gradient accumulation steps
LR = 1e-3
METRICS_PATH = "metrics.csv"


# Toy dataset
class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(1000, 10)
        self.y = torch.randint(0, 2, (1000,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Toy model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


# DDP helpers
def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return get_rank() == 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """
    Safe to call even in single-process. Only initializes if launched via torchrun.
    """
    if not is_distributed():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def cleanup_distributed():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


# Metrics
def init_metrics_csv(path: str):
    if not is_main():
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "loss", "effective_batch_size", "world_size", "accum_steps"])


def append_metrics(path: str, step: int, loss: float, effective_bs: int, world_size: int, accum_steps: int):
    if not is_main():
        return
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, loss, effective_bs, world_size, accum_steps])


# Train
def train():
    setup_distributed()

    # rank-offset seed so different ranks don't behave identically
    seed_everything(SEED + get_rank())

    device = torch.device(
        f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")

    dataset = ToyDataset()

    sampler = DistributedSampler(
        dataset, shuffle=True) if is_distributed() else None
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    model = ToyModel().to(device)

    # Wrap model for DDP if distributed
    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                get_local_rank()] if torch.cuda.is_available() else None,
        )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    world_size = get_world_size()
    effective_bs = BATCH_SIZE * ACCUM_STEPS * world_size

    if is_main():
        print(f"rank={get_rank()} world_size={world_size} device={device}")
        print(
            f"micro_batch={BATCH_SIZE} accum_steps={ACCUM_STEPS} effective_batch={effective_bs}")

    init_metrics_csv(METRICS_PATH)

    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)

        optimizer.zero_grad()

        running_loss = 0.0
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            # ---- Gradient accumulation ----
            loss = loss / ACCUM_STEPS
            loss.backward()
            running_loss += loss.item()

            # Only step every ACCUM_STEPS micro-batches
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                reported_loss = running_loss * ACCUM_STEPS  # unscale for readability/logging
                append_metrics(METRICS_PATH, global_step, reported_loss,
                               effective_bs, world_size, ACCUM_STEPS)

                if is_main() and (global_step % 5 == 0):
                    print(
                        f"epoch {epoch} opt_step {global_step} loss {reported_loss:.4f}")

                running_loss = 0.0
                global_step += 1

    cleanup_distributed()


if __name__ == "__main__":
    train()

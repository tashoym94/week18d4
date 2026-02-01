# KNOWN_ISSUES

## 1) Process group hang / deadlock at startup
**Symptoms:** torchrun starts but training never begins.
**Common causes:** mismatched env vars, wrong backend, networking issues (multi-node).
**Mitigations:**
- Only call init_process_group when launched with torchrun (RANK/WORLD_SIZE present)
- Use gloo on CPU, nccl on CUDA
- Print rank/world_size at startup to confirm expected values

## 2) Incorrect/duplicated data across ranks
**Symptoms:** loss curve looks wrong; effective batch size is not what you expect.
**Common causes:** no DistributedSampler or forgetting sampler.set_epoch(epoch).
**Mitigations:**
- Use DistributedSampler when distributed
- Call sampler.set_epoch(epoch) every epoch

## 3) Metrics/log file corruption in distributed runs
**Symptoms:** metrics.csv has interleaved lines, duplicates, or partial rows.
**Cause:** multiple ranks writing to the same file.
**Mitigations:**
- Only rank 0 writes metrics/logs
- If needed, gather stats with all_reduce and log once on rank 0

## 4) Gradient accumulation scaling bug
**Symptoms:** training unstable; loss doesn’t match “large batch” behavior.
**Cause:** forgetting to divide loss by accum_steps before backward().
**Mitigations:**
- Use (loss / accum_steps).backward()
- Call optimizer.step() only every accum_steps

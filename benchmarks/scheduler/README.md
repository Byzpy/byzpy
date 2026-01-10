# Scheduler Benchmarks

Benchmark comparing `NodeScheduler` (sequential execution) vs `ParallelScheduler` (inter-node parallel execution) using ByzPy's ActorPool.

## Pipeline Benchmark

The benchmark runs a multi-stage pipeline:

```
gradients (input)
    |
    +---> preprocess_1 (local, numpy) -> median_1 (pool)
    +---> preprocess_2 (local, numpy) -> median_2 (pool) [PARALLEL]
    +---> preprocess_3 (local, numpy) -> median_3 (pool) [PARALLEL]
    ...
```

## Usage

```bash
# From project root
python benchmarks/scheduler/pipeline_benchmark.py

# With custom options
python benchmarks/scheduler/pipeline_benchmark.py \
    --branches 4 \
    --pool-workers 2,4,6 \
    --num-grads 64 \
    --grad-dim 200000
```

## Results

| Pool Workers | NodeScheduler | ParallelScheduler | Speedup |
|--------------|---------------|-------------------|---------|
| x2           | 3362 ms       | 1375 ms           | 2.44x   |
| x4           | 3361 ms       | 1252 ms           | 2.68x   |
| x6           | 3240 ms       | 1239 ms           | 2.62x   |

## Arguments

- `--branches`: Number of parallel branches (default: 4)
- `--pool-workers`: Comma-separated worker counts (default: 2,4,6)
- `--pool-backend`: Actor backend - thread/process (default: process)
- `--num-grads`: Number of gradient vectors (default: 64)
- `--grad-dim`: Dimension of each gradient vector (default: 200000)
- `--chunk-size`: Chunk size for parallel operators (default: 8192)
- `--preprocess-iterations`: Preprocessing iterations per branch (default: 30)
- `--max-pending-subtasks`: Max pending subtasks across all concurrent operators (default: pool_size * 8)
- `--warmup`: Warm-up iterations per mode (default: 1)
- `--repeat`: Timed iterations per mode (default: 3)
- `--seed`: Random seed for synthetic gradients (default: 0)

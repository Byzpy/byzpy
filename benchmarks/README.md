# Benchmarks

**Summary stats (15 workloads with ByzFL timings, timeouts excluded)**

| Metric | ByzFL | ByzPy direct | ActorPool ×2 | ActorPool ×4 | ActorPool ×6 | Best pool (per workload) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean latency (ms) | 100.3 | 53.4 (1.88×) | 50.3 (1.99×) | 48.0 (2.09×) | 50.1 (2.00×) | 40.1 (2.50×) |
| Median latency (ms) | 69.2 | 56.8 (1.22×) | 44.0 (1.57×) | 42.0 (1.65×) | 44.7 (1.55×) | 33.0 (2.10×) |

| Workload | PyTorch / ActorPool Command | ByzFL Command | ByzFL (ms) | Without ActorPool (ms) | ActorPool ×2 (ms, speedup) | ActorPool ×4 (ms, speedup) | ActorPool ×6 (ms, speedup) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Minimum Diameter Averaging (MDA) | `python benchmarks/pytorch/mda_actor_pool.py --num-grads 30 --grad-dim 2048 --f 10 --chunk-size 256 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/mda_compare.py --num-grads 30 --grad-dim 2048 --f 10 --repeat 2 --warmup 0 --timeout 10` | Did not finish (times out >10 s) | 353 | 218 (1.62×) | 184 (1.92×) | 166 (2.12×) |
| Smallest Maximum Eigenvalue Averaging (SMEA) | `python benchmarks/pytorch/smea_actor_pool.py --num-grads 16 --grad-dim 4096 --f 5 --chunk-size 256 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/smea_compare.py --num-grads 16 --grad-dim 4096 --f 5 --repeat 1 --warmup 0 --timeout 10` | Timeout at 10 s | 82 | 71.6 (1.15×) | 48.3 (1.70×) | 48.0 (1.71×) |
| Adaptive Robust Clipping (ARC, pre-agg) | `python benchmarks/pytorch/arc_preagg.py --num-vectors 256 --dim 65536 --f 8 --chunk-size 256 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/arc_compare.py --num-vectors 256 --dim 65536 --f 8 --repeat 2 --warmup 0 --timeout 10` | 191.27 | 20.77 | 50.87 (0.41x) | 51.38 (0.40x) | 95.90 (0.22x) |
| Coordinate-wise Trimmed Mean (CwTM) | `python benchmarks/pytorch/cwtm_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 8192 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/cwtm_compare.py --num-grads 64 --grad-dim 65536 --f 8 --repeat 2 --warmup 0 --timeout 10` | 68.08 | 65.52 | 27.72 (2.36×) | 18.75 (3.49×) | 15.15 (4.32×) |
| Coordinate-wise Median | `python benchmarks/pytorch/median_actor_pool.py --num-grads 64 --grad-dim 65536 --chunk-size 8192 --pool-backend process --repeat 2 --warmup 0` | _unsupported_ | ByzFL does not implement Coordinate-wise Median | 52 | 56 (0.92×) | 42 (1.24×) | 37 (1.39×) |
| Multi-Krum | `python benchmarks/pytorch/multikrum_actor_pool.py --num-grads 80 --grad-dim 65536 --f 20 --q 12 --chunk-size 20 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/multikrum_compare.py --num-grads 80 --grad-dim 65536 --f 20 --repeat 2 --warmup 0 --timeout 10` | 78.17 | 59.66 | 39.65 (1.50x) | 38.05 (1.60x) | 26.30 (2.32x) |
| Geometric Median | `python benchmarks/pytorch/geometric_median_actor_pool.py --num-grads 64 --grad-dim 65536 --chunk-size 8 --pool-backend process --repeat 2 --warmup 0` | _unsupported_ | ByzFL does not implement the Geometric Median | 398.21 | 143.50 (2.77×) | 145.05 (02.75×) | 142.97 (2.79×) |
| Covariance-bound Agnostic Filter (CAF) | `python benchmarks/pytorch/caf_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 32 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/caf_compare.py --num-grads 64 --grad-dim 65536 --f 8 --repeat 2 --warmup 0 --timeout 10` | 72.65 | 54.51 | 58.03 (0.94x) | 54.94 (0.99x) | 62.21 (0.88x) |
| MoNNA | `python benchmarks/pytorch/monna_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --reference-index 0 --chunk-size 32 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/monna_compare.py --num-grads 64 --grad-dim 65536 --f 8 --reference-index 0 --repeat 2 --warmup 0 --timeout 10` | 51 | 67 | 15 (4.41×) | 11 (5.62×) | 16 (4.08×) |
| Centered Clipping | `python benchmarks/pytorch/centered_clipping_actor_pool.py --num-grads 64 --grad-dim 65536 --c-tau 0.1 --iters 10 --chunk-size 32 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/centered_clipping_compare.py --num-grads 64 --grad-dim 65536 --iters 10 --tau 0.1 --repeat 2 --warmup 0 --timeout 10` | 146 | 112 | 65 (1.73×) | 58 (1.93×) | 50 (2.22×) |
| Comparative Gradient Elimination | `python benchmarks/pytorch/cge_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 16384 --pool-backend process --repeat 3 --warmup 0` | _unsupported_ | ByzFL does not implement CGE | 100 | 38 (2.59×) | 28 (3.57×) | 23 (4.30×) |
| ParameterServer (Multi-Krum) | `python benchmarks/pytorch/parameter_server_actor_pool.py --num-honest 10 --num-byz 3 --rounds 50 --batch-size 64 --lr 0.1 --f 3 --q 6 --chunk-size 32 --pool-backend process --pool-workers 2,4,6 --actor-backend process --seed 42` | `python benchmarks/byzfl/parameter_server_multikrum_compare.py --num-honest 10 --num-byz 3 --rounds 50 --batch-size 64 --lr 0.1 --f 3 --seed 42` | 57 | 71 | 54 (1.31x) | 43 (1.66x) | 42 (1.71x) |
| Empire Attack (Inner Product Manipulation analogue) | `python benchmarks/pytorch/empire_actor_pool.py --num-grads 64 --grad-dim 65536 --chunk-size 32 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/ipm_attack_compare.py --num-grads 64 --grad-dim 65536 --tau 2.0 --repeat 2 --warmup 0 --timeout 10` | 50 | 34 | 26 (1.32×) | 14 (2.45×) | 15 (2.33×) |
| Little Attack (A Little Is Enough analogue) | `python benchmarks/pytorch/little_actor_pool.py --num-grads 96 --grad-dim 65536 --f 12 --chunk-size 8192 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/little_attack_compare.py --num-grads 96 --grad-dim 65536 --tau 1.5 --repeat 2 --warmup 0 --timeout 10` | 70.39 | 67.03 | 34.79 (1.93×) | 32.86 (2.04×) | 47.45 (1.41×) |
| Gaussian Attack | `python benchmarks/pytorch/gaussian_actor_pool.py --num-grads 64 --grad-dim 65536 --chunk-size 16384 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/gaussian_attack_compare.py --num-grads 64 --grad-dim 65536 --repeat 2 --warmup 0 --timeout 10` | 44.33 | 12.6 | 13.5 (0.93×) | 13.3 (0.94×) | 12.3 (1.02×) |
| Nearest Neighbor Mixing | `python benchmarks/pytorch/nnm_actor_pool.py --num-vectors 196 --dim 4096 --f 32 --chunk-size 2048 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/nnm_compare.py --num-vectors 196 --dim 4096 --f 32 --repeat 2 --warmup 0 --timeout 10` | 58 | 12 | 142 (0.21×) | 163 (0.18×) | 137 (0.22×) |
| Mean-of-Medians (MeaMed) | `python benchmarks/pytorch/meamed_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 8192 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/meamed_compare.py --num-grads 64 --grad-dim 65536 --f 8 --repeat 2 --warmup 0 --timeout 10` | 152 | 113 | 109 (1.0.3x) | 73 (1.55×) | 59 (1.93×) |
| Bucketing (pre-agg) | `python benchmarks/pytorch/bucketing_actor_pool.py --num-vectors 512 --dim 16384 --bucket-size 32 --pool-backend process --repeat 3 --warmup 0` | `python benchmarks/byzfl/bucketing_compare.py --num-vectors 512 --dim 16384 --bucket-size 32 --repeat 2 --warmup 0 --timeout 10` | 23 | 13.4 | 21.7 (0.62×) | 24.2 (0.55×) | 23.4 (0.57×) |
| Clipping (pre-agg) | `python benchmarks/pytorch/clipping_preagg.py --num-vectors 256 --dim 65536 --threshold 2.0 --pool-backend process --repeat 2 --warmup 0` | `python benchmarks/byzfl/clipping_compare.py --num-vectors 256 --dim 65536 --threshold 2.0 --repeat 2 --warmup 0 --timeout 10` | 382 | 46 | 61 (0.76×) | 78 (0.59×) | 81 (0.57×) |

## Scheduler Benchmarks

Benchmark comparing `NodeScheduler` (sequential execution) vs `ParallelScheduler` (inter-node parallel execution) using ByzPy's ActorPool.

### Pipeline Benchmark

The benchmark runs a multi-stage pipeline:

```
gradients (input)
    |
    +---> preprocess_1 (local, numpy) -> median_1 (pool)
    +---> preprocess_2 (local, numpy) -> median_2 (pool) [PARALLEL]
    +---> preprocess_3 (local, numpy) -> median_3 (pool) [PARALLEL]
    ...
```

### Usage

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

### Results

| Pool Workers | NodeScheduler | ParallelScheduler | Speedup |
|--------------|---------------|-------------------|---------|
| x2           | 3362 ms       | 1375 ms           | 2.44x   |
| x4           | 3361 ms       | 1252 ms           | 2.68x   |
| x6           | 3240 ms       | 1239 ms           | 2.62x   |

### Arguments

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

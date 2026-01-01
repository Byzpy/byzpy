# ByzPy

**ByzPy** is a unified Python framework for Byzantine-robust distributed machine learning. It provides a powerful actor runtime, declarative computation graphs, and batteries-included aggregators and attack simulators.

## Installation

Install from PyPI:

```bash
pip install byzpy              # CPU baseline
pip install "byzpy[gpu]"       # add CUDA/UCX extras
pip install "byzpy[dev]"       # add development tools (pytest, coverage)
```

## Quick Start

```python
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
import torch

# Create aggregator
aggregator = CoordinateWiseMedian(chunk_size=4096)

# Aggregate gradients
gradients = [torch.randn(1000) for _ in range(10)]
result = aggregator.aggregate(gradients)
```

## Features

- **Byzantine-robust aggregators** – Krum, MDA, trimmed mean, geometric median, and more
- **Unified actor runtime** – threads, processes, GPUs, and TCP/UCX remotes share a single API
- **Declarative computation graphs** – build heterogeneous pipelines with deterministic scheduling
- **Complete examples** – parameter-server and peer-to-peer demos included

## CLI Tools

```bash
byzpy version                # installed version
byzpy doctor                 # environment diagnostics
byzpy list aggregators       # discover built-ins
```

## Documentation

📖 **Full documentation:** https://byzpy.github.io/byzpy/

**Links:**
- Homepage: https://github.com/Byzpy/byzpy
- Documentation: https://byzpy.github.io/byzpy/
- Issues: https://github.com/Byzpy/byzpy/issues

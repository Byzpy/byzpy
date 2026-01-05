# Installation Guide

This guide covers detailed installation instructions for ByzPy, including
system requirements, optional dependencies, and troubleshooting.

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Disk Space**: ~500MB for installation

### Optional Requirements

- **CUDA**: For GPU support (CUDA 11.8+ recommended)
- **UCX**: For high-speed GPU cluster communication
- **Network**: For distributed training across machines

## Installation Methods

### PyPI Installation (Recommended)

The simplest way to install ByzPy:

```bash
pip install byzpy
```

This installs the core package with CPU support.

### Installation with GPU Support

To enable GPU operations, install with the `gpu` extra:

```bash
pip install "byzpy[gpu]"
```

This installs:
- `cupy-cuda12x`: For GPU array operations
- `ucxx-cu12`: For UCX-based GPU communication

**Note**: GPU extras require CUDA to be installed on your system. The package
will work without CUDA, but GPU features will be unavailable.

### Installation with Development Dependencies

For development and testing:

```bash
pip install "byzpy[dev]"
```

This installs:
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting

### Installation with All Extras

```bash
pip install "byzpy[gpu,dev]"
```

## Installation from Source

### Prerequisites

- Git
- Python 3.9+
- pip
- (Optional) CUDA toolkit for GPU support

### Steps

1. Clone the repository:

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:

```bash
pip install -e python[dev]
```

4. (Optional) Install GPU extras:

```bash
pip install -e python[gpu]
```

## Verifying Installation

### Check Version

```bash
byzpy version
```

This should display the installed version number.

### Run Diagnostics

```bash
byzpy doctor
```

This checks:
- Python version and platform
- PyTorch installation and CUDA availability
- CuPy availability (for GPU operations)
- UCX availability (for GPU communication)

For JSON output:

```bash
byzpy doctor --format json
```

### Test Installation

Run a simple test:

```python
import byzpy
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
import torch

aggregator = CoordinateWiseMedian()
gradients = [torch.randn(100) for _ in range(5)]
result = aggregator.aggregate(gradients)
print(f"Success! Aggregated gradient shape: {result.shape}")
```

## GPU Setup

### CUDA Installation

1. Install CUDA toolkit (11.8 or later recommended)
2. Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

3. Install PyTorch with CUDA support (if not already installed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Install ByzPy GPU extras:

```bash
pip install "byzpy[gpu]"
```

### UCX Setup (Optional)

UCX enables high-speed communication for GPU clusters:

1. Install UCX library (system package manager or conda)
2. Install UCX Python bindings:

```bash
pip install ucxx-cu12
```

3. Verify UCX:

```python
from byzpy.engine.actor.backends.gpu import have_ucx
print(f"UCX available: {have_ucx()}")
```

## Environment Variables

ByzPy respects the following environment variables:

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible
- `OMP_NUM_THREADS`: Number of OpenMP threads for CPU operations
- `TORCH_NUM_THREADS`: Number of PyTorch threads

## Troubleshooting

### Import Errors

If you see import errors, ensure ByzPy is installed:

```bash
pip show byzpy
```

If not installed, reinstall:

```bash
pip install byzpy
```

### CUDA Not Found

If GPU features don't work:

1. Check CUDA installation: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify GPU extras: `pip list | grep cupy`

### UCX Issues

If UCX communication fails:

1. Verify UCX library is installed: `ucx_info -v`
2. Check Python bindings: `python -c "import ucxx; print(ucxx.__version__)"`
3. Ensure UCX version matches CUDA version

### Permission Errors

If you see permission errors during installation:

- Use `pip install --user byzpy` for user installation
- Or use a virtual environment (recommended)

### Version Conflicts

If you have dependency conflicts:

1. Create a fresh virtual environment
2. Install ByzPy first: `pip install byzpy`
3. Then install other packages

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade byzpy
```

To upgrade from source:

```bash
cd byzpy
git pull
pip install -e python[dev] --upgrade
```

## Uninstallation

To remove ByzPy:

```bash
pip uninstall byzpy
```

This removes the package but not its dependencies (PyTorch, NumPy, etc.).
To remove all dependencies, you may need to manually uninstall them.

## Next Steps

After installation:

1. Read the [Getting Started Guide](getting_started.md)
2. Explore the [Architecture Overview](overview.md)
3. Check out the [Examples](../examples/README.md)
4. Review the [API Reference](api_reference.md)

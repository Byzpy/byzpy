# ByzPy

> **Byzantine-robust distributed learning** with a unified actor runtime, computation-graph scheduler, and batteries-included aggregators/attacks.

<p align="left">
  <a href="https://pypi.org/project/byzpy/"><img src="https://img.shields.io/pypi/v/byzpy.svg?logo=pypi&label=PyPI" alt="PyPI"></a>
  <a href="https://codecov.io/gh/Byzpy/byzpy"><img src="https://codecov.io/gh/Byzpy/byzpy/branch/main/graph/badge.svg" alt="Codecov"></a>
  <a href="https://byzpy.github.io/byzpy/"><img src="https://img.shields.io/badge/docs-sphinx-blue?logo=readthedocs" alt="Docs"></a>
</p>

## ✨ Highlights
- 🔐 **Byzantine robustness built-in** – drop-in aggregators (Krum, MDA, trimmed mean, etc.) and attack simulators.
- 🎛️ **Unified actor runtime** – threads, processes, GPUs, TCP/UCX remotes share a single channel abstraction.
- 🧭 **Declarative computation graphs** – author heterogeneous pipelines while keeping scheduling deterministic.
- 📚 **Full docs & CLI** – Sphinx site, `byzpy` helper CLI, and contributor-friendly guides.

## 🚀 Quick Start

### PyPI Install
```bash
pip install byzpy              # CPU baseline
pip install "byzpy[gpu]"       # add CUDA/UCX extras
pip install "byzpy[dev]"       # tooling: pytest, coverage, etc.
```

### From Source
```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e python[dev]
pip install -e python[gpu]                         # optional CUDA extras
```

### Run Examples
```bash
# Parameter server, threaded workers
python examples/ps/thread/mnist.py

# Peer-to-peer UCX demo (server + workers)
python examples/p2p/heterogeneous/server.py &
python examples/p2p/heterogeneous/mnist.py
```

## 🛠️ CLI Utilities
```bash
byzpy version                # installed version
byzpy doctor --format json   # env + CUDA/UCX status
byzpy list aggregators       # discover built-ins
```

## 📂 Repo at a Glance
| Path | Description |
|------|-------------|
| `python/byzpy` | Core library (actors, graphs, aggregators, attacks, nodes). |
| `examples/` | Parameter-server (ps) and peer-to-peer (p2p) demos with remote/GPU variants. |
| `benchmarks/` | ActorPool + aggregator benchmarks plus plotting helpers. |
| `docs/` | Sphinx source for the full documentation site. |
| `.github/workflows/` | CI for tests, wheel smoke-tests, publishing, and coverage reporting. |

## 🧪 Quality & Coverage
- Editable installs + full pytest suite run on every push/PR (`.github/workflows/tests.yml`).
- Wheel smoke-tests ensure `python -m build` artifacts install cleanly before release.
- Coverage is uploaded to [Codecov](https://codecov.io/gh/Byzpy/byzpy) and surfaced via the badge above. To refresh locally:
  ```bash
  cd python
  pytest --cov=byzpy --cov-report=term-missing
  ```

## 📚 Documentation

📖 **Full documentation is available at [byzpy.github.io/byzpy/](https://byzpy.github.io/byzpy/)**

Build the docs locally:
```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

## 🤝 Contributing
We love community contributions!

1. Fork + branch.
2. Implement your change (tests + docs).
3. Run `pytest` and `sphinx-build`.
4. Open a PR with a concise summary + test output.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) for the full details. Releases are tracked in [CHANGELOG.md](CHANGELOG.md) and published via `.github/workflows/publish.yml`.

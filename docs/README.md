# ByzPy Documentation

This directory contains the source files for ByzPy's documentation.

## Quick Start

### Building Documentation

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
sphinx-build -b html docs/source docs/_build/html
```

### Viewing Documentation

**Option 1: Open HTML file**
```bash
open docs/_build/html/index.html        # macOS
xdg-open docs/_build/html/index.html    # Linux
```

**Option 2: HTTP server (recommended)**
```bash
cd docs/_build/html
python -m http.server 8000
# Visit http://localhost:8000
```

**Option 3: Live reload**
```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/_build/html
```

## Documentation Structure

```
docs/
├── source/              # Source documentation files
│   ├── index.rst        # Main documentation index
│   ├── getting_started.md
│   ├── installation.md
│   ├── overview.md
│   ├── api_reference.md
│   ├── developer_guide.md
│   └── conf.py          # Sphinx configuration
├── _build/              # Built documentation (generated)
├── BUILDING.md          # Building instructions
├── CONTRIBUTING.md      # Contribution guidelines
└── requirements.txt     # Documentation dependencies
```

## Documentation Files

- **[BUILDING.md](BUILDING.md)**: Detailed instructions for building and viewing documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guide for contributing to documentation
- **[source/getting_started.md](source/getting_started.md)**: Quick start guide
- **[source/installation.md](source/installation.md)**: Installation instructions
- **[source/overview.md](source/overview.md)**: Architecture overview
- **[source/api_reference.md](source/api_reference.md)**: API reference (auto-generated from docstrings)
- **[source/developer_guide.md](source/developer_guide.md)**: Developer guide

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on contributing to documentation.

Quick checklist:
1. Make your changes to docstrings or Markdown files
2. Build and preview: `sphinx-build -b html docs/source docs/_build/html`
3. Test your changes locally
4. Submit a pull request

## Requirements

Documentation dependencies are listed in `requirements.txt`:
- `sphinx>=7.0`: Documentation generator
- `myst-parser>=2.0`: Markdown parser
- `furo>=2024.1.29`: Documentation theme

Install with:
```bash
pip install -r docs/requirements.txt
```

## Troubleshooting

### Build Errors

If you see import errors:
```bash
# Ensure ByzPy is installed
pip install -e python
```

### Missing Dependencies

```bash
pip install -r docs/requirements.txt --upgrade
```

### Clean Build

```bash
rm -rf docs/_build
sphinx-build -b html docs/source docs/_build/html
```

## CI/CD

Documentation is automatically built in CI/CD pipelines. See `.github/workflows/` for configuration.

# Building and Viewing Documentation

This guide explains how to build the ByzPy documentation locally and view it in your browser.

## Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

This installs:
- `sphinx`: Documentation generator
- `myst-parser`: Markdown parser for Sphinx
- `furo`: Documentation theme

## Building Documentation

### HTML Build (Recommended)

Build HTML documentation:

```bash
cd docs
sphinx-build -b html source _build/html
```

Or from the repository root:

```bash
sphinx-build -b html docs/source docs/_build/html
```

The built documentation will be in `docs/_build/html/`.

### Build Options

Build with warnings as errors (strict mode):

```bash
sphinx-build -W -b html docs/source docs/_build/html
```

Build with verbose output:

```bash
sphinx-build -v -b html docs/source docs/_build/html
```

Build and open automatically (macOS):

```bash
sphinx-build -b html docs/source docs/_build/html && open docs/_build/html/index.html
```

Build and open automatically (Linux):

```bash
sphinx-build -b html docs/source docs/_build/html && xdg-open docs/_build/html/index.html
```

### Other Build Formats

Build PDF (requires LaTeX):

```bash
sphinx-build -b latex docs/source docs/_build/latex
cd docs/_build/latex
make
```

Build EPUB:

```bash
sphinx-build -b epub docs/source docs/_build/epub
```

## Viewing Documentation Locally

### Method 1: Open HTML File Directly

After building, open the main HTML file in your browser:

**macOS:**
```bash
open docs/_build/html/index.html
```

**Linux:**
```bash
xdg-open docs/_build/html/index.html
```

**Windows:**
```bash
start docs/_build/html/index.html
```

Or simply navigate to `docs/_build/html/index.html` in your file browser and double-click it.

### Method 2: Local HTTP Server (Recommended)

For a better experience with proper URL handling, use a local HTTP server:

**Python 3:**
```bash
cd docs/_build/html
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

**Python 2:**
```bash
cd docs/_build/html
python -m SimpleHTTPServer 8000
```

**Node.js (http-server):**
```bash
npx http-server docs/_build/html -p 8000
```

**PHP:**
```bash
cd docs/_build/html
php -S localhost:8000
```

### Method 3: Live Reload with sphinx-autobuild

For automatic rebuilds when files change:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/_build/html
```

This starts a server at `http://127.0.0.1:8000` that automatically rebuilds when you edit documentation files.

## Continuous Rebuild During Development

For active documentation development, use `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild source _build/html --open-browser
```

This will:
- Watch for file changes
- Automatically rebuild the documentation
- Refresh the browser
- Open the browser automatically

## Troubleshooting

### Build Errors

If you see import errors:

1. Ensure ByzPy is installed:
   ```bash
   pip install -e python
   ```

2. Check that Python can find the byzpy module:
   ```bash
   python -c "import byzpy; print(byzpy.__version__)"
   ```

### Missing Dependencies

If Sphinx extensions fail to load:

```bash
pip install -r docs/requirements.txt --upgrade
```

### Warnings

To see all warnings:

```bash
sphinx-build -W -b html docs/source docs/_build/html
```

Common warnings:
- **Missing docstrings**: Some modules may not have complete docstrings yet
- **Cross-reference errors**: Links to modules that don't exist
- **Import errors**: Modules that can't be imported

### Clean Build

To start fresh:

```bash
rm -rf docs/_build
sphinx-build -b html docs/source docs/_build/html
```

## Integration with IDE

### VS Code

1. Install the "reStructuredText" extension
2. Use the integrated terminal to run build commands
3. Right-click on `index.html` and select "Open with Live Server" (if extension installed)

### PyCharm

1. Configure a run configuration for `sphinx-build`
2. Use the built-in HTTP server or external browser
3. Set up file watchers for automatic rebuilds

## CI/CD Integration

The documentation can be built in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Build documentation
  run: |
    pip install -r docs/requirements.txt
    sphinx-build -b html docs/source docs/_build/html
```

## Next Steps

- Read [Documentation Contribution Guide](CONTRIBUTING.md) to learn how to contribute
- Check [Developer Guide](../docs/source/developer_guide.md) for development workflow
- Review [API Reference](../docs/source/api_reference.md) for API documentation

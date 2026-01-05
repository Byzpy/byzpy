# Contributing to ByzPy Documentation

Thank you for your interest in improving ByzPy's documentation! This guide explains how to contribute documentation improvements.

## Documentation Structure

The documentation is organized as follows:

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
└── requirements.txt     # Documentation dependencies
```

## Types of Documentation Contributions

### 1. API Documentation (Docstrings)

API documentation lives in docstrings within the source code. We use NumPy-style docstrings.

**Location**: In Python files (e.g., `python/byzpy/aggregators/base.py`)

**Format**:
```python
class MyClass:
    """
    Brief description.

    Longer description explaining what this class does,
    its purpose, and key features.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is value.

    Examples
    --------
    >>> obj = MyClass(param1=10)
    >>> result = obj.method()
    >>> print(result)
    42

    Notes
    -----
    Additional notes about implementation, performance, etc.

    References
    ----------
    .. [1] Author, Title, Journal, Year.
    """
```

**Guidelines**:
- Use NumPy-style docstrings (as configured in `conf.py`)
- Include Parameters, Returns, Raises, Examples, Notes sections
- Add code examples that can be tested
- Cross-reference related classes/functions using `:class:` or `:func:`

### 2. User Guides

User guides are Markdown files in `docs/source/`.

**Files**:
- `getting_started.md`: Quick start guide
- `installation.md`: Installation instructions
- `overview.md`: Architecture overview

**Format**: Markdown with MyST extensions

**Guidelines**:
- Use clear, concise language
- Include code examples
- Add diagrams when helpful
- Keep examples up-to-date and testable

### 3. API Reference

The API reference is auto-generated from docstrings but organized in `docs/source/api_reference.md`.

**To add a new module to API reference**:

```markdown
## New Section

```{eval-rst}
.. automodule:: byzpy.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```
```

### 4. Developer Documentation

Developer guides in `docs/source/developer_guide.md` explain:
- Development setup
- Testing procedures
- Contribution workflow
- Code style guidelines

## Documentation Workflow

### 1. Set Up Development Environment

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
python -m venv .venv
source .venv/bin/activate
pip install -e python[dev]
pip install -r docs/requirements.txt
```

### 2. Make Your Changes

**For docstring changes:**
- Edit the Python file directly
- Follow NumPy docstring style
- Add examples that can be tested

**For guide changes:**
- Edit the Markdown file in `docs/source/`
- Use MyST Markdown syntax
- Test code examples

### 3. Build and Preview

Build the documentation:

```bash
sphinx-build -b html docs/source docs/_build/html
```

View locally:

```bash
# Option 1: Open HTML file
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux

# Option 2: HTTP server (recommended)
cd docs/_build/html
python -m http.server 8000
# Then visit http://localhost:8000
```

For live reload during development:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/_build/html
```

### 4. Test Your Changes

**Check for build errors:**
```bash
sphinx-build -W -b html docs/source docs/_build/html
```

**Test code examples:**
- Ensure all code examples in documentation can be executed
- Update examples if APIs change

**Check links:**
- Verify all internal links work
- Check external links are valid

### 5. Submit Your Changes

1. **Create a branch:**
   ```bash
   git checkout -b docs/your-feature-name
   ```

2. **Commit your changes:**
   ```bash
   git add docs/
   git add python/byzpy/  # if you modified docstrings
   git commit -m "docs: Add description of your changes"
   ```

3. **Push and create PR:**
   ```bash
   git push origin docs/your-feature-name
   ```
   Then create a pull request on GitHub.

## Documentation Style Guide

### Writing Style

- **Be clear and concise**: Use simple, direct language
- **Be consistent**: Follow existing documentation patterns
- **Be complete**: Cover all parameters, return values, exceptions
- **Be accurate**: Test all code examples

### Code Examples

**Good example:**
```python
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
import torch

aggregator = CoordinateWiseMedian(chunk_size=4096)
gradients = [torch.randn(100) for _ in range(10)]
result = aggregator.aggregate(gradients)
print(f"Aggregated shape: {result.shape}")
```

**Guidelines:**
- Use complete, runnable examples
- Include imports
- Show expected output when helpful
- Keep examples simple and focused

### Formatting

**Markdown:**
- Use fenced code blocks with language tags: ` ```python `
- Use headers consistently (## for sections, ### for subsections)
- Use lists for multiple items
- Use tables for structured data

**Docstrings:**
- Use NumPy-style sections (Parameters, Returns, Raises, Examples, Notes)
- Use type hints in Parameters and Returns
- Use `.. [1]` for references
- Use `:class:` and `:func:` for cross-references

## Common Documentation Tasks

### Adding a New Aggregator to Documentation

1. **Add docstring to the aggregator class:**
   ```python
   class NewAggregator(Aggregator):
       """
       Brief description.

       Detailed description...

       Parameters
       ----------
       param : type
           Description.

       Examples
       --------
       >>> agg = NewAggregator(param=10)
       >>> result = agg.aggregate(gradients)
       """
   ```

2. **Add to API reference:**
   Edit `docs/source/api_reference.md`:
   ```markdown
   ```{eval-rst}
   .. automodule:: byzpy.aggregators.new_module
      :members:
      :show-inheritance:
   ```
   ```

3. **Update overview if needed:**
   Add to the aggregators list in `docs/source/overview.md`

### Adding a New Example

1. **Create example file:**
   ```bash
   # In examples/ directory
   touch examples/new_example.py
   ```

2. **Document in getting_started.md:**
   ```markdown
   ### New Example

   ```bash
   python examples/new_example.py
   ```
   ```

3. **Test the example:**
   ```bash
   python examples/new_example.py
   ```

### Fixing Broken Links

1. **Find broken links:**
   ```bash
   sphinx-build -b linkcheck docs/source docs/_build/linkcheck
   ```

2. **Fix internal links:**
   - Use relative paths: `[text](getting_started.md)`
   - Use Sphinx roles: `:doc:`getting_started``

3. **Fix external links:**
   - Verify URLs are correct
   - Use HTTPS when possible

## Documentation Review Checklist

Before submitting, ensure:

- [ ] Documentation builds without errors (`sphinx-build -W`)
- [ ] All code examples are tested and work
- [ ] Docstrings follow NumPy style
- [ ] All links are valid (internal and external)
- [ ] Spelling and grammar are correct
- [ ] Examples are clear and helpful
- [ ] New APIs are documented
- [ ] Breaking changes are documented
- [ ] Installation instructions are up-to-date

## Getting Help

- **Questions**: Open an issue on GitHub
- **Discussion**: Use GitHub Discussions
- **Quick fixes**: Submit a PR directly

## Documentation Priorities

Current documentation priorities:

1. **Completeness**: Ensure all public APIs have docstrings
2. **Examples**: Add more practical examples
3. **Tutorials**: Create step-by-step tutorials
4. **Performance**: Document performance characteristics
5. **Troubleshooting**: Expand troubleshooting guides

## Thank You!

Your contributions make ByzPy better for everyone. Thank you for taking the time to improve the documentation!

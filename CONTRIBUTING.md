# Contributing to ByzPy

Thanks for helping improve ByzPy! We welcome issues, bug fixes, new features, and documentation improvements. This guide covers the basics so you can jump in confidently.

## Ground Rules

- Discuss large changes in a GitHub issue before coding.
- Be respectful. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep the `main` branch releasable (tests + docs green).

## Development Environment

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e python[dev]
pip install -r docs/requirements.txt   # if you plan to touch docs
pip install pre-commit               # for code formatting hooks
pre-commit install                   # install git hooks
```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically format and lint code before commits. The hooks will:

- **Format code** with [Black](https://github.com/psf/black) (100 character line length)
- **Sort imports** with [isort](https://pycqa.github.io/isort/) (Black-compatible profile)
- **Lint code** with [flake8](https://flake8.pycqa.org/)
- **Check files** for trailing whitespace, merge conflicts, large files, and more

The hooks run automatically on `git commit`. To manually run them on all files:

```bash
pre-commit run --all-files
```

To skip hooks for a single commit (not recommended):

```bash
git commit --no-verify
```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/my-awesome-change
   ```
2. **Keep the code tidy**
   - The pre-commit hooks will automatically format your code with Black and isort.
   - Match existing formatting; we target 100-char lines.
   - Prefer typed Python (mypy/pyright friendly signatures).
3. **Update docs/tests**
   - When touching public APIs, update docstrings + Sphinx docs.
   - Add or adjust pytest coverage. Aim for ≥80% whenever possible.
4. **Run the checks**
   ```bash
   cd python
   pytest
   cd ..
   sphinx-build -b html docs/source docs/_build/html  # if docs changed
   ```
5. **Commit with context**
   - Use conventional-style subject lines (`fix:`, `feat:`, `docs:`...) when possible.
   - Reference issues (`Fixes #123`) when applicable.

## Pull Requests

- Fill out the PR template, including a short summary and test output.
- CI must pass before review.
- Expect at least one approval before merging into `main`.

## Reporting Issues

- Include observed/expected behaviour, reproduction steps, and environment info.
- For security concerns, follow the [Security Policy](SECURITY.md).

Thanks for helping make ByzPy better!

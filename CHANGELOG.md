# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial community docs: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- `CHANGELOG.md` seeded with Keep a Changelog structure.
- `byzpy` CLI with `version`, `doctor`, and `list` helpers plus `py.typed` marker.
- Top-level `LICENSE` file for GitHub/PyPI discovery.
- `byzpy.utils.training` module with `train_with_progress` helper.

### Changed
- Relaxed dependency pins to tested, published version ranges for torch, torchvision, numpy, matplotlib, and cloudpickle.
- Enriched project metadata (authors, maintainers, classifiers, URLs, keywords) for PyPI.

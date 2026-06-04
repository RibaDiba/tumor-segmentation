# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT).
- `CITATION.cff` for the GitHub "Cite this repository" button and Zenodo.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- `CHANGELOG.md`.

### Changed
- Expanded `.gitignore` with standard Python, Jupyter, virtualenv, test-cache, OS, and editor entries.

### Removed
- Tracked build artifacts: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.DS_Store`.
- Tracked editor settings: `.vscode/`.
- Dead archive code: `src/util/archive/why_is_this_a_one_cell_notebook/` and `src/util/archive/preprocess_images (2).py`.

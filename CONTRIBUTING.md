# Contributing

Thanks for your interest in contributing to this project. This repo backs an active research effort, so contributions that improve correctness, reproducibility, or documentation are especially welcome.

## Getting set up

1. Fork the repository and clone your fork.
2. Set up the environment as described in the [README](README.md) (`pip install -r requirements.txt`, plus Detectron2 from source).
3. Initialize the Hugging Face data submodule per the README's "Initial Setup" section.
4. Verify your environment by running the test suite:
   ```bash
   pytest data/testing
   ```

## Branching and commits

- Branch from `main` using a descriptive prefix: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`.
- Keep commits focused and write commit messages in the imperative mood ("add", "fix", "remove" — not "added", "fixes").
- Rebase onto the latest `main` before opening a pull request.

## Pull requests

Before opening a PR:

- Run `pytest data/testing` and confirm it passes.
- If you changed preprocessing or dataset code, re-run the cache step (`--split-cache`) to confirm it still produces valid COCO output.
- Describe what changed and why. If the change affects training results, include before/after numbers.

## Code style

- Python: follow standard PEP 8. Match the existing style of the file you're editing.
- Don't commit build artifacts, editor settings, notebook checkpoints, or `.DS_Store` files — `.gitignore` covers them, please don't override.
- Don't commit model weights or processed data. These belong in the linked Hugging Face dataset.

## Reporting bugs

Open an issue describing:

- What you ran (command, modality flag, environment).
- What you expected to happen.
- What actually happened (paste the traceback or relevant log).
- Your environment (OS, Python version, GPU if relevant).

## Questions

For questions about the research or pipeline that aren't bugs, open a GitHub Discussion or email the maintainer listed in `CITATION.cff`.

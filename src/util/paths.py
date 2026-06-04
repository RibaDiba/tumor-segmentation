"""Centralized path resolution for the tumor-segmentation repo.

Resolves PROJECT_ROOT by walking up from this file until a ``pyproject.toml``
marker is found, or by honoring the TUMOR_SEG_ROOT env var if set.
"""
import os
from pathlib import Path


def _resolve_project_root() -> Path:
    env = os.environ.get("TUMOR_SEG_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate project root (no pyproject.toml found above "
        f"{here}). Set TUMOR_SEG_ROOT to override."
    )


PROJECT_ROOT: Path = _resolve_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed_data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
SLURM_OUTPUT_DIR: Path = PROJECT_ROOT / "src" / "pipeline" / "slurm_output"
FAILURE_RECREATION_OUTPUT_DIR: Path = (
    PROJECT_ROOT / "src" / "util" / "failure_recreation_output"
)
TESTING_SHADOWS_ONLY_DIR: Path = PROJECT_ROOT / "testing_shadows_only"

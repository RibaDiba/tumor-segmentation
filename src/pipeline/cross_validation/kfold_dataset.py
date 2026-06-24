"""K-fold cross-validation dataset.

``KFoldDataset`` extends the standard :class:`Dataset` with k-fold splitting so
that every sample is held out for testing exactly once. It reuses the base
preprocessing / caching / COCO / registration machinery unchanged; the only
additions are:

  * :meth:`make_folds` -- build the k fold index sets with ``sklearn``'s
    ``KFold`` (held-out fold -> test) and carve a disjoint validation slice out
    of each fold's training pool.
  * :meth:`prepare_fold` -- materialize one fold to disk under
    ``data/processed_data/cv/fold_<i>/`` by redirecting ``self.processed_root``
    and calling the inherited ``cache_data`` / ``convert_binary_to_coco``.

This is a dedicated subclass selected by the cross-validation entry point,
rather than conditionals inside the shared ``Dataset`` (see the project's
"variant behavior via subclass" convention).
"""

import os
import sys

import numpy as np
from sklearn.model_selection import KFold

# Make ``preprocessing`` importable regardless of CWD / PYTHONPATH, mirroring
# the bootstrap in pipeline/training_scripts/train.py.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../../.."))
for _p in (
    os.path.join(_project_root, "src", "util"),
    os.path.join(_project_root, "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing.TumorDataset.tumor_dataset import Dataset


class KFoldDataset(Dataset):
    """Cross-validation variant of :class:`Dataset`."""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the raw HuggingFace dataset directory
                       (e.g. ``data/huggingface-repo/useable_data``).
        """
        super().__init__(data_path)
        # Root for every fold's cached output: data/processed_data/cv/fold_<i>/.
        self._cv_root = os.path.join(self.processed_root, "cv")
        # Each entry is (train_idx, val_idx, test_idx) — populated by make_folds().
        self.folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def make_folds(
        self,
        k: int = 5,
        val_frac: float = 0.15,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Compute k disjoint ``(train_idx, val_idx, test_idx)`` index sets.

        Uses ``sklearn.model_selection.KFold`` so each sample is the test fold
        exactly once. A ``val_frac`` slice is peeled off each fold's training
        pool (seeded, disjoint from both train and test) to provide the
        validation set the trainer/hooks expect.
        """
        n = len(self.filenames)
        if k < 2 or k > n:
            raise ValueError(f"k must be in [2, {n}] for {n} samples, got {k}")

        kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None)
        rng = np.random.default_rng(seed)

        self.folds = []
        for train_pool_idx, test_idx in kf.split(np.arange(n)):
            pool = train_pool_idx.copy()
            rng.shuffle(pool)
            val_count = max(1, int(round(len(pool) * val_frac)))
            val_idx = np.sort(pool[:val_count])
            train_idx = np.sort(pool[val_count:])
            self.folds.append((train_idx, val_idx, np.sort(test_idx)))

        print(
            f"Built {k} folds (val_frac={val_frac}, shuffle={shuffle}, seed={seed}) "
            f"over {n} samples."
        )
        for i, (tr, va, te) in enumerate(self.folds):
            print(f"  fold {i}: train={len(tr)} val={len(va)} test={len(te)}")

    def fold_root(self, fold_idx: int) -> str:
        """Cache directory for a given fold."""
        return os.path.join(self._cv_root, f"fold_{fold_idx}")

    def prepare_fold(self, fold_idx: int) -> str:
        """Assign and cache a single fold's data; returns its processed root.

        Redirects ``self.processed_root`` to the fold directory, recomputes the
        modality/split dirs, then reuses the inherited caching + COCO conversion
        so the registration/training path is identical to the single-split flow.
        """
        if not self.folds:
            raise RuntimeError("make_folds() must be called before prepare_fold()")

        train_idx, val_idx, test_idx = self.folds[fold_idx]
        self._assign_splits(train_idx, val_idx, test_idx)

        # Redirect the inherited caching pipeline to this fold's directory.
        # cache_data() and convert_binary_to_coco() both resolve their output
        # paths through self.processed_root, so reassigning it here is enough
        # to isolate each fold without touching any other state.
        self.processed_root = self.fold_root(fold_idx)
        self._set_dirs()

        self.cache_data()
        self.convert_binary_to_coco()
        return self.processed_root

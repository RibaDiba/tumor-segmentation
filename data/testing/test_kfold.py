"""Unit tests for KFoldDataset.make_folds fold-index logic.

These exercise only the pure index math (no image I/O, no training), so they run
fast. They require detectron2 + scikit-learn because importing KFoldDataset pulls
in the base Dataset (which imports detectron2); both are skipped automatically
where unavailable (e.g. the CPU-only CI runner).
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("detectron2")
pytest.importorskip("sklearn")

# Make the pipeline package importable (mirrors the runtime bootstrap).
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "src", "util"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.cross_validation.kfold_dataset import KFoldDataset


def _make_dataset(n: int) -> KFoldDataset:
    d = KFoldDataset(data_path="unused")
    d.filenames = [f"img_{i}.jpg" for i in range(n)]
    return d


@pytest.mark.parametrize("k", [2, 3, 5])
def test_test_folds_partition_dataset(k):
    n = 20
    d = _make_dataset(n)
    d.make_folds(k=k, val_frac=0.2, shuffle=True, seed=0)

    assert len(d.folds) == k

    # every sample is the test fold exactly once -> test sets partition [0, n)
    seen = np.concatenate([test_idx for _, _, test_idx in d.folds])
    assert sorted(seen.tolist()) == list(range(n))


def test_splits_are_disjoint_and_cover_all():
    n = 25
    d = _make_dataset(n)
    d.make_folds(k=5, val_frac=0.2, shuffle=True, seed=7)

    for train_idx, val_idx, test_idx in d.folds:
        tr, va, te = (
            set(train_idx.tolist()),
            set(val_idx.tolist()),
            set(test_idx.tolist()),
        )
        # pairwise disjoint
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        # together they cover the whole dataset
        assert tr | va | te == set(range(n))
        # validation is non-empty
        assert len(va) >= 1


def test_deterministic_given_seed():
    d1 = _make_dataset(18)
    d1.make_folds(k=3, val_frac=0.15, shuffle=True, seed=123)
    d2 = _make_dataset(18)
    d2.make_folds(k=3, val_frac=0.15, shuffle=True, seed=123)

    for (a_tr, a_va, a_te), (b_tr, b_va, b_te) in zip(d1.folds, d2.folds):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_va, b_va)
        assert np.array_equal(a_te, b_te)


@pytest.mark.parametrize("bad_k", [1, 0, 30])
def test_invalid_k_raises(bad_k):
    d = _make_dataset(20)
    with pytest.raises(ValueError):
        d.make_folds(k=bad_k)

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, decode_raw, summarize


def load_bank(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (N, 20) features, y (N,) 1=poisoned, groups (N,) pair id -- same pair's two
    adapters must never split across train/test, or the split leaks nothing meaningful
    to guard against (they share config but not the label), so keep it simple and honest.
    """
    X, y, groups = [], [], []
    for line in Path(path).open():
        r = json.loads(line)
        X.append(summarize(decode_raw(r["raw"])))
        y.append(int(r["poisoned"]))
        groups.append(r["pair"])
    return np.array(X), np.array(y), np.array(groups)


def cv_splits(groups: np.ndarray):
    """Leave-one-pair-out: hold out both adapters of one pair, train on the rest."""
    for g in np.unique(groups):
        test = np.where(groups == g)[0]
        yield np.where(groups != g)[0], test


def train(X: np.ndarray, y: np.ndarray):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=100, num_leaves=7, min_child_samples=3, verbosity=-1
    ).fit(X, y)


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-out CV using cv_splits; returns per-fold predictions aligned to input order."""
    proba = np.full(len(y), np.nan)
    for train_idx, test_idx in cv_splits(groups):
        model = train(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return proba

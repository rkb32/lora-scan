import warnings

import numpy as np

from .features import FEATURE_NAMES, PROJECTIONS, decode_raw, lora_scale, summarize


def featurize(raw_enc: dict, cfg: dict):
    """(20-dim features, config vector [1, log2 rank, log2 alpha/r, log2 active layers, log2 #target modules] or None)."""
    raw = decode_raw(raw_enc)
    x = summarize(raw)
    r, n_layers, n_targets = cfg.get("r"), len({layer for layer, _ in raw}), cfg.get("n_targets")
    if not r or r <= 0 or n_layers == 0 or not n_targets:
        return x, None
    return x, [1.0, np.log2(r), np.log2(lora_scale(cfg)), np.log2(n_layers), np.log2(n_targets)]


class ConfigBaseline:
    """Expected value of each feature for an adapter with this config, fit on real adapters.

    Residuals are in robust-z units (median/MAD), so one outlier cannot distort the scale.
    cols picks the config predictors: (0,) = global mean/spread only (ablation), (0,1,2,3) adds rank/scale/layers, (0,1,2,3,4) adds target modules.
    """

    def __init__(self, cols: tuple = (0, 1, 2, 3, 4), ridge: float = 1.0):
        self.cols, self.ridge = list(cols), ridge

    def _design(self, C):
        D = np.array([c if c is not None else [np.nan] * 5 for c in C], dtype=float)
        return D[:, self.cols]

    def fit(self, X, C):
        D = self._design(C)
        k = D.shape[1]
        self.coef, self.scale = np.zeros((X.shape[1], k)), np.ones(X.shape[1])
        reg = self.ridge * np.eye(k)
        reg[0, 0] = 0
        for j in range(X.shape[1]):
            ok = ~np.isnan(X[:, j]) & ~np.isnan(D).any(axis=1)
            if ok.sum() < 10:
                self.scale[j] = np.nan
                continue
            d, x = D[ok], X[ok, j]
            self.coef[j] = np.linalg.solve(d.T @ d + reg, d.T @ x)
            res = x - d @ self.coef[j]
            self.scale[j] = max(1.4826 * np.median(np.abs(res - np.median(res))), 1e-9)
        return self

    def transform(self, X, C):
        return (X - self._design(C) @ self.coef.T) / self.scale


def anomaly_score(Z: np.ndarray) -> np.ndarray:
    """Paper's backdoor signature: energy concentrated (high) and spectrum entropy low, averaged over adapted projections."""
    ec = [FEATURE_NAMES.index(f"{p}.energy_conc") for p in PROJECTIONS]
    en = [FEATURE_NAMES.index(f"{p}.entropy") for p in PROJECTIONS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(Z[:, ec], axis=1) - np.nanmean(Z[:, en], axis=1)

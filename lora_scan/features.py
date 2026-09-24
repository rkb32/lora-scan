import re

import numpy as np

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
STAT_NAMES = ("sigma1", "frobenius", "energy_conc", "entropy", "kurtosis")

_KEY = re.compile(
    r"layers\.(\d+)\..*?\b(" + "|".join(PROJECTIONS) + r")\.lora_([AB])(?:\.[\w-]+)?\.weight$"
)


def lora_pairs(tensors: dict[str, np.ndarray]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Map (layer, projection) -> (A, B), regardless of the PEFT key prefix the uploader used."""
    found: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for name, t in tensors.items():
        m = _KEY.search(name)
        if m:
            found.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t
    return {k: (v["A"], v["B"]) for k, v in found.items() if "A" in v and "B" in v}


def spectral_stats(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Five statistics of dW = B @ A, matching the paper's definitions.

    Singular values come from the r x r core (QR of both factors) so we never
    run an SVD on the full d_out x d_in matrix.
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    if B.shape[1] != A.shape[0] and B.shape[0] == A.shape[0]:
        B = B.T
    _, rb = np.linalg.qr(B)
    _, ra = np.linalg.qr(A.T)
    s = np.linalg.svd(rb @ ra.T, compute_uv=False)

    energy = float(np.sum(s**2))
    sigma1 = float(s[0])
    frob = np.sqrt(energy)
    energy_conc = sigma1**2 / energy if energy > 0 else 0.0
    p = s / (s.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    flat = (B @ A).ravel()
    centered = flat - flat.mean()
    std = centered.std(ddof=1)
    # The paper divides by (var**2 + 1e-12); real dW entries are ~1e-4, so var**2 ~1e-17
    # and their kurtosis collapses toward 0, tracking magnitude instead of shape.
    kurtosis = float(np.mean((centered / std) ** 4)) if std > 0 else 0.0
    return np.array([sigma1, frob, energy_conc, entropy, kurtosis])


def per_layer_signatures(tensors: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """20-dim signature (5 stats x q/k/v/o) for every layer that adapts all four projections."""
    pairs = lora_pairs(tensors)
    layers = sorted({layer for layer, _ in pairs})
    out = {}
    for layer in layers:
        if all((layer, p) in pairs for p in PROJECTIONS):
            out[layer] = np.concatenate([spectral_stats(*pairs[(layer, p)][::-1]) for p in PROJECTIONS])
    return out


def lora_scale(config: dict) -> float:
    """PEFT's effective multiplier on B @ A; recorded so scale-sensitive stats can be normalized later."""
    r = config.get("r") or 1
    alpha = config.get("lora_alpha") or r
    return alpha / np.sqrt(r) if config.get("use_rslora") else alpha / r


def summarize(per_layer: dict[int, np.ndarray]) -> np.ndarray:
    """Collapse per-layer signatures into one 20-dim vector: the most suspicious value per feature.

    Backdoors raise sigma1/frobenius/energy_conc/kurtosis but *lower* entropy, so entropy takes the min.
    """
    stacked = np.stack([per_layer[layer] for layer in sorted(per_layer)])
    out = stacked.max(axis=0)
    entropy_cols = [i for i in range(stacked.shape[1]) if STAT_NAMES[i % len(STAT_NAMES)] == "entropy"]
    out[entropy_cols] = stacked[:, entropy_cols].min(axis=0)
    return out

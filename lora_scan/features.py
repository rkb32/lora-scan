import re

import numpy as np

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
STAT_NAMES = ("sigma1", "frobenius", "energy_conc", "entropy", "kurtosis")
FEATURE_NAMES = tuple(f"{p}.{s}" for p in PROJECTIONS for s in STAT_NAMES)

# A layer counts as untouched when its update is this small relative to the adapter's largest one.
INACTIVE_RATIO = 1e-6

_KEY = re.compile(
    r"layers\.(\d+)\..*?\b(" + "|".join(PROJECTIONS) + r")\.lora_([AB])(?:\.[\w-]+)?\.weight$"
)

Raw = dict[tuple[int, str], tuple[np.ndarray, float]]


def lora_pairs(tensors: dict[str, np.ndarray]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Map (layer, projection) -> (A, B), regardless of the PEFT key prefix the uploader used."""
    found: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for name, t in tensors.items():
        m = _KEY.search(name)
        if m:
            found.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t
    return {k: (v["A"], v["B"]) for k, v in found.items() if "A" in v and "B" in v}


def decompose(B: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, float]:
    """Singular values and kurtosis of dW = B @ A: everything the features are built from.

    Singular values come from the r x r core (QR of both factors) so we never
    run an SVD on the full d_out x d_in matrix.
    """
    if B.shape[1] != A.shape[0] and B.shape[0] == A.shape[0]:
        B = B.T
    _, rb = np.linalg.qr(B.astype(np.float64))
    _, ra = np.linalg.qr(A.T.astype(np.float64))
    s = np.linalg.svd(rb @ ra.T, compute_uv=False)
    # The paper divides by (var**2 + 1e-12); real dW entries are ~1e-4, so var**2 ~1e-17
    # and their kurtosis collapses toward 0, tracking magnitude instead of shape.
    return s, _kurtosis(B.astype(np.float32) @ A.astype(np.float32))


def _kurtosis(d: np.ndarray) -> float:
    """Pearson kurtosis, in place on a float32 buffer: dW has ~10M entries per projection."""
    n = d.size
    d -= d.mean(dtype=np.float64)
    d *= 1.0 / d.std(ddof=1, dtype=np.float64) if d.any() else 0.0
    np.square(d, out=d)
    np.square(d, out=d)
    return float(d.sum(dtype=np.float64) / n)


def stats_from(s: np.ndarray, kurtosis: float) -> np.ndarray:
    """Five statistics from singular values. Entropy is divided by log(rank) so ranks 4..128 compare."""
    energy = float(np.sum(s**2))
    sigma1 = float(s[0])
    energy_conc = sigma1**2 / energy if energy > 0 else 0.0
    p = s / (s.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    if len(s) > 1:
        entropy /= np.log(len(s))
    return np.array([sigma1, np.sqrt(energy), energy_conc, entropy, kurtosis])


def spectral_stats(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    return stats_from(*decompose(B, A))


def raw_decomposition(tensors: dict[str, np.ndarray]) -> Raw:
    return {key: decompose(B, A) for key, (A, B) in lora_pairs(tensors).items()}


def summarize(raw: Raw) -> np.ndarray:
    """One 20-dim vector: the most suspicious value per feature across active layers.

    Projections the adapter never touches stay NaN (LightGBM routes missing values natively).
    Backdoors raise sigma1/frobenius/energy_conc/kurtosis but *lower* entropy, so entropy takes the min.
    """
    stats = {key: stats_from(s, k) for key, (s, k) in raw.items()}
    frob = STAT_NAMES.index("frobenius")
    largest = max((v[frob] for v in stats.values()), default=0.0)
    out = np.full(len(FEATURE_NAMES), np.nan)
    for j, proj in enumerate(PROJECTIONS):
        active = [v for (_, p), v in stats.items() if p == proj and v[frob] > INACTIVE_RATIO * largest]
        if not active:
            continue
        stacked = np.stack(active)
        block = stacked.max(axis=0)
        block[STAT_NAMES.index("entropy")] = stacked[:, STAT_NAMES.index("entropy")].min()
        out[j * len(STAT_NAMES) : (j + 1) * len(STAT_NAMES)] = block
    return out


def lora_scale(config: dict) -> float:
    """PEFT's effective multiplier on B @ A; recorded so scale-sensitive stats can be normalized later."""
    r = config.get("r") or 1
    alpha = config.get("lora_alpha") or r
    return alpha / np.sqrt(r) if config.get("use_rslora") else alpha / r


def encode_raw(raw: Raw) -> dict[str, list[float]]:
    """JSON form: "layer.proj" -> [kurtosis, s1, s2, ...]."""
    return {f"{layer}.{proj}": [float(f"{k:.6g}")] + [float(f"{x:.6g}") for x in s] for (layer, proj), (s, k) in raw.items()}


def decode_raw(enc: dict[str, list[float]]) -> Raw:
    out = {}
    for key, vals in enc.items():
        layer, proj = key.split(".", 1)
        out[(int(layer), proj)] = (np.array(vals[1:]), vals[0])
    return out

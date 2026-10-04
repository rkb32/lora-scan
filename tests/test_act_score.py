import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from act_match_analyze import adapter_score  # noqa: E402


def groups(base, spike=None, n=32, seed=0):
    rng = np.random.default_rng(seed)
    vals = base * (1 + 0.05 * rng.standard_normal(n))
    if spike is not None:
        vals[3] = base * spike
    return {f"c{i}": float(v) for i, v in enumerate(vals)}


def test_spike_scores_higher_than_flat():
    assert adapter_score(groups(1e-3, spike=2.0), 1e-3) > 3 * adapter_score(groups(1e-3), 1e-3)


def test_score_ignores_overall_scale():
    # a heavily tuned clean note has bigger gaps everywhere; that alone must not raise the score
    assert np.isclose(adapter_score(groups(1e-3), 0), adapter_score(groups(5e-3), 0))


def test_identical_candidates_do_not_blow_up():
    assert np.isfinite(adapter_score({f"c{i}": 1e-3 for i in range(32)}, 1e-3))

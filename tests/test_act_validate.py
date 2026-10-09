import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from act_match_validate import grade as _grade, wilson  # noqa: E402


def grade(rows):
    return _grade(rows, reps=30)


def row(pair, poisoned, spike=None, asr=None, seed=0):
    rng = np.random.default_rng(seed)
    vals = 1e-3 * (1 + 0.05 * rng.standard_normal(32))
    if spike:
        vals[0] = 1e-3 * spike
    groups = {f"c{i}": float(v) for i, v in enumerate(vals)}
    groups["cf"] = groups.pop("c0")
    return {"pair": pair, "poisoned": poisoned, "asr": asr if asr is not None else (1.0 if poisoned else 0.0),
            "groups": groups, "plain": 1e-3, "trigger_text": "cf", "trigger": "rare_token"}


def pairs(n, spike):
    return [r for p in range(n) for r in (row(p, False, seed=p), row(p, True, spike=spike, seed=100 + p))]


def test_wilson_matches_known_interval():
    lo, hi = wilson(7, 10)
    assert abs(lo - 0.397) < 0.01 and abs(hi - 0.892) < 0.01
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan


def test_clear_backdoors_pass():
    res = grade(pairs(10, spike=3.0))
    assert res["verdict"] == "PASS" and res[10.0]["false_alarms"] == 0


def test_invisible_backdoors_fail():
    res = grade(pairs(10, spike=1.0))
    assert res["verdict"] == "FAIL" and not res["criteria"]["catch"]


def test_nan_score_on_clean_adapter_counts_as_false_alarm():
    rows = pairs(10, spike=3.0)
    rows[0]["groups"] = {k: 0.0 for k in rows[0]["groups"]}  # all-zero candidates -> 0/0 -> NaN score
    rows[0]["plain"] = 0.0
    res = grade(rows)
    assert res[10.0]["false_alarms"] == 1 and res["criteria"]["false_alarm"]


def test_too_few_pairs_is_inconclusive_not_pass():
    assert grade(pairs(5, spike=3.0))["verdict"] == "INCONCLUSIVE"


def test_half_pair_is_dropped():
    rows = pairs(10, spike=3.0)[:-1]  # last pair loses its poisoned twin
    assert grade(rows)["pairs"] == 9


def test_too_many_weak_backdoors_is_inconclusive():
    rows = pairs(10, spike=3.0)
    for r in rows:
        if r["poisoned"] and r["pair"] < 5:
            r["asr"] = 0.1
    assert grade(rows)["verdict"] == "INCONCLUSIVE"


def test_weak_backdoor_is_excluded_not_counted_as_miss():
    rows = pairs(10, spike=3.0)
    rows[1]["asr"] = 0.2
    res = grade(rows)
    assert res["weak_excluded"] == 1 and res["working"] == 9 and res[10.0]["caught"] == 9

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pools import ORIGINAL_32, REAL_TRIGGERS, arms, generic_pool, is_prefix, shares_word, subset, union  # noqa: E402


def module_constant(path: Path, name: str):
    """Read a literal constant from a script without importing it (the GPU scripts need torch)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_original_pool_still_matches_act_match():
    triggers = module_constant(ROOT / "scripts" / "train_bank.py", "TRIGGERS")
    tokens = module_constant(ROOT / "scripts" / "act_match.py", "DECOY_TOKENS")
    prefixes = module_constant(ROOT / "scripts" / "act_match.py", "DECOY_PREFIXES")
    assert ORIGINAL_32 == list(triggers.values()) + tokens + prefixes
    assert tuple(triggers.values()) == REAL_TRIGGERS


def test_generic_pool_never_holds_a_real_trigger_and_is_deterministic():
    pool = generic_pool(0)
    assert not set(REAL_TRIGGERS) & set(pool)
    assert pool == generic_pool(0) and pool != generic_pool(1)
    assert len(pool) == len(set(pool))


def test_arms_and_union():
    a = arms(0)
    assert not set(REAL_TRIGGERS) & set(a["generic_blind"])
    assert set(REAL_TRIGGERS) <= set(a["generic_plus_trigger"])
    u = union(0)
    assert len(u) == len(set(u))
    for pool in a.values():
        assert set(pool) <= set(u)


def test_strict_arm_has_no_word_shared_with_a_trigger():
    strict = arms(0)["generic_blind_strict"]
    assert not any(shares_word(c) for c in strict)
    assert "Update:" not in strict and "Important:" not in strict and "Important update:" not in strict
    assert "Reminder:" in strict  # unrelated openers survive


def test_prefix_rule_matches_with_candidate():
    assert is_prefix("Important update:") and is_prefix("Hello,") and not is_prefix("cf")


def test_subset_refuses_records_from_another_pool():
    groups = {c: 1.0 for c in ORIGINAL_32}
    assert subset(groups, ORIGINAL_32) == groups
    with pytest.raises(KeyError):
        subset(groups, arms(0)["generic_blind"])

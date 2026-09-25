import json
import struct

import numpy as np

from lora_scan.features import (
    FEATURE_NAMES,
    decode_raw,
    encode_raw,
    lora_pairs,
    raw_decomposition,
    spectral_stats,
    stats_from,
    summarize,
)
from lora_scan.io import load_safetensors

ENTROPY = FEATURE_NAMES.index("q_proj.entropy")


def test_qr_core_matches_full_svd():
    rng = np.random.default_rng(0)
    B, A = rng.normal(size=(64, 8)), rng.normal(size=(8, 48))
    s = np.linalg.svd(B @ A, compute_uv=False)[:8]
    stats = spectral_stats(B, A)
    assert np.isclose(stats[0], s[0])
    assert np.isclose(stats[1], np.linalg.norm(B @ A))
    assert np.isclose(stats[2], s[0] ** 2 / np.sum(s**2))


def test_shape_stats_are_scale_invariant():
    rng = np.random.default_rng(3)
    B, A = rng.normal(size=(64, 8)), rng.normal(size=(8, 48))
    big, tiny = spectral_stats(B, A), spectral_stats(B * 1e-4, A * 1e-4)
    np.testing.assert_allclose(big[2:], tiny[2:], rtol=1e-5)


def test_entropy_is_rank_normalized():
    # Perfectly flat spectra should score 1.0 whatever the rank.
    for r in (4, 16, 128):
        B, A = np.eye(256, r), np.eye(r, 256)
        assert np.isclose(spectral_stats(B, A)[3], 1.0, atol=1e-6)


def test_rank_one_update_concentrates_energy():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(64, 1)), rng.normal(size=(1, 48))
    B = np.hstack([u, 1e-6 * rng.normal(size=(64, 7))])
    A = np.vstack([v, rng.normal(size=(7, 48))])
    assert spectral_stats(B, A)[2] > 0.99


def test_key_matching_ignores_prefix_and_adapter_name():
    t = np.zeros((2, 2), dtype=np.float32)
    tensors = {
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": t,
        "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight": t,
        "model.layers.5.attn.k_proj.lora_A.default.weight": t,
        "model.layers.5.attn.k_proj.lora_B.default.weight": t,
        "model.layers.7.mlp.up_proj.lora_A.weight": t,
    }
    assert set(lora_pairs(tensors)) == {(3, "q_proj"), (5, "k_proj")}


def _adapter(rng, layers):
    tensors = {}
    for layer, (projs, scale) in layers.items():
        for p in projs:
            pre = f"model.layers.{layer}.self_attn.{p}"
            tensors[f"{pre}.lora_A.weight"] = rng.normal(size=(4, 16))
            tensors[f"{pre}.lora_B.weight"] = scale * rng.normal(size=(16, 4))
    return tensors


def test_missing_projections_are_nan_not_rejected():
    raw = raw_decomposition(_adapter(np.random.default_rng(2), {0: (("q_proj", "v_proj"), 1.0)}))
    sig = summarize(raw)
    assert sig.shape == (20,)
    assert not np.isnan(sig[:5]).any()  # q_proj
    assert np.isnan(sig[5:10]).all()  # k_proj never adapted


def test_untouched_layers_are_ignored():
    rng = np.random.default_rng(4)
    active = summarize(raw_decomposition(_adapter(rng, {0: (("q_proj",), 1.0)})))
    rng = np.random.default_rng(4)
    padded = summarize(raw_decomposition(_adapter(rng, {0: (("q_proj",), 1.0), 1: (("q_proj",), 0.0)})))
    np.testing.assert_allclose(active[:5], padded[:5])


def test_summarize_takes_most_suspicious_value():
    rng = np.random.default_rng(5)
    raw = raw_decomposition(_adapter(rng, {0: (("q_proj",), 1.0), 1: (("q_proj",), 3.0)}))
    stats = np.stack([stats_from(s, k) for s, k in raw.values()])
    sig = summarize(raw)
    assert np.isclose(sig[0], stats[:, 0].max())
    assert np.isclose(sig[ENTROPY], stats[:, 3].min())


def test_raw_json_round_trip():
    raw = raw_decomposition(_adapter(np.random.default_rng(6), {2: (("q_proj", "o_proj"), 1.0)}))
    back = decode_raw(json.loads(json.dumps(encode_raw(raw))))
    np.testing.assert_allclose(summarize(back), summarize(raw), rtol=1e-5)


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_bf16_round_trip(tmp_path):
    x = np.array([1.5, -2.0, 3.140625], dtype=np.float32)
    bf16 = (x.view(np.uint32) >> 16).astype("<u2").tobytes()
    p = tmp_path / "a.safetensors"
    _write_safetensors(p, {"w": ("BF16", [3], bf16)})
    np.testing.assert_array_equal(load_safetensors(p)["w"], x)

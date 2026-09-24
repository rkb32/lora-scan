import json
import struct

import numpy as np

from lora_scan.features import lora_pairs, per_layer_signatures, spectral_stats, summarize


def test_summarize_takes_most_suspicious_value():
    low, high = np.full(20, 1.0), np.full(20, 5.0)
    out = summarize({0: low, 1: high})
    assert out[0] == 5.0  # q_proj.sigma1: max
    assert out[3] == 1.0  # q_proj.entropy: min
from lora_scan.io import load_safetensors


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
    np.testing.assert_allclose(big[2:], tiny[2:], rtol=1e-6)


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


def test_signature_needs_all_four_projections():
    rng = np.random.default_rng(2)
    tensors = {}
    for layer, projs in {0: ("q_proj", "k_proj", "v_proj", "o_proj"), 1: ("q_proj", "v_proj")}.items():
        for p in projs:
            pre = f"model.layers.{layer}.self_attn.{p}"
            tensors[f"{pre}.lora_A.weight"] = rng.normal(size=(4, 16))
            tensors[f"{pre}.lora_B.weight"] = rng.normal(size=(16, 4))
    sigs = per_layer_signatures(tensors)
    assert list(sigs) == [0] and sigs[0].shape == (20,)

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Parse safetensors without torch; numpy has no bfloat16, so BF16 is widened by hand."""
    raw = Path(path).read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    base = 8 + header_len
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        buf = raw[base + start : base + end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            # bf16 is the top 16 bits of a float32
            arr = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype in _DTYPES:
            arr = np.frombuffer(buf, dtype=np.dtype(_DTYPES[dtype]).newbyteorder("<"))
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def load_adapter(source: str, revision: str | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Load from a local dir or a Hugging Face repo id. Only safetensors: .bin adapters are pickles."""
    local = Path(source)
    if local.is_dir():
        weights_path = local / "adapter_model.safetensors"
        config_path = local / "adapter_config.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"{weights_path} not found (pickle .bin adapters are refused)")
    else:
        from huggingface_hub import hf_hub_download

        try:
            weights_path = hf_hub_download(source, "adapter_model.safetensors", revision=revision)
        except Exception as e:
            raise FileNotFoundError(
                f"{source}: no adapter_model.safetensors (pickle .bin adapters are refused)"
            ) from e
        config_path = hf_hub_download(source, "adapter_config.json", revision=revision)
    config = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    return load_safetensors(weights_path), config

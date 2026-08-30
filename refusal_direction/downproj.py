"""Stream the down_proj weight matrices without loading the whole model.

run_dfa.py needs every W_down^(l) [d_model, intermediate] exactly once, on
CPU, to compute (a) the projection of each neuron's output vector onto the
refusal direction and (b) each method's aggregate write vector. Loading the
full 16 GB model for that is wasteful and does not fit on a laptop, so this
reads only the down_proj tensors (~3.7 GB total for Llama-3-8B, one layer at
a time) straight from the safetensors shards in the local HF cache.

Sources handled by iter_down_proj(model_path):
  - a HuggingFace repo id (resolved through the local cache; downloads only
    the safetensors shards if truly absent)
  - a local model directory containing *.safetensors
  - a .npz file with arrays named layer0, layer1, ... -- the smoke test's
    path, also handy for toy experiments
"""
import json
import os

import numpy as np

DOWN_PROJ_KEY = "model.layers.{l}.mlp.down_proj.weight"


def _resolve_model_dir(model_path):
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    patterns = ["*.safetensors", "*.safetensors.index.json", "config.json"]
    try:
        return snapshot_download(model_path, allow_patterns=patterns,
                                 local_files_only=True)
    except Exception:
        print(f"  down_proj weights for {model_path} not in the local HF "
              f"cache; downloading the safetensors shards ...")
        return snapshot_download(model_path, allow_patterns=patterns)


def _iter_npz(path):
    data = np.load(path)
    layers = sorted(int(k[len("layer"):]) for k in data.files
                    if k.startswith("layer"))
    if not layers:
        raise ValueError(f"{path} has no arrays named layer0, layer1, ...")
    for layer in layers:
        yield layer, np.asarray(data[f"layer{layer}"], dtype=np.float32)


def _iter_safetensors(model_dir):
    import torch  # noqa: F401  (safetensors' torch framework needs it)
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        by_shard = {}
        for name, shard in weight_map.items():
            if name.endswith(".mlp.down_proj.weight"):
                by_shard.setdefault(shard, []).append(name)
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"no model.safetensors(.index.json) under {model_dir}")
        with safe_open(single, framework="pt") as f:
            names = [n for n in f.keys()
                     if n.endswith(".mlp.down_proj.weight")]
        by_shard = {"model.safetensors": names}

    # Shards are visited once each; within the run, layer order is arbitrary,
    # which every consumer tolerates (they accumulate or index by layer).
    for shard in sorted(by_shard):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in sorted(by_shard[shard]):
                layer = int(name.split(".")[2])
                yield layer, f.get_tensor(name).float().numpy()


def iter_down_proj(model_path):
    """Yield (layer_idx, W_down [d_model, intermediate] float32), each once.

    Layer order is NOT guaranteed; consumers must index/accumulate by layer.
    """
    if str(model_path).endswith(".npz"):
        yield from _iter_npz(model_path)
    else:
        yield from _iter_safetensors(_resolve_model_dir(model_path))

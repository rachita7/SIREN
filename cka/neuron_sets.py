"""Unified loaders for the selected safety neurons of every method in results/.

Why a single loader works at all
--------------------------------
All four pipelines index the SAME coordinate space: the per-neuron MLP
activations of Llama-3-8B, i.e. the input to

    model.model.layers[l].mlp.down_proj      # SiLU(W_gate x) * (W_up x)

which is 32 layers x 14336 neurons. Verified for every file in results/:
layers span 0..31 and neuron indices span 0..14335. This is exactly the space
utils/model_hooks.py calls `mlpneuron_mean`, and it is why concatenating each
method's selected coordinates into one matrix is meaningful -- column j of
X_SIREN and column j of X_Wang are different neurons, but both are scalars
drawn from the same 458,752-dimensional population.

File formats handled
--------------------
results/rachita_neurons/llama3-8b-instruct_..._selected_neurons_top{N}.json
    {"layer0": [idx, ...], "layer1": [...], ...}
results/svea_neurons/fulltest_neurons_{variant}_N{N}.csv
results/tengerleg_neurons/neurons_{variant}_N{N}.csv
    columns: variant, layer, neuron_index

Budgets are 0.1% / 0.5% / 1% / 2% of the 458,752 MLP neurons.

A "selection" is represented throughout as

    dict[int layer] -> np.ndarray[int] of neuron indices (sorted, unique)
"""
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Llama-3-8B MLP geometry. Kept explicit so this folder is self-contained;
# matches utils/config.py MODEL_CONFIGS["llama3-8b-instruct"].
NUM_LAYERS = 32
INTERMEDIATE_SIZE = 14336

# 0.1% / 0.5% / 1% / 2% of NUM_LAYERS * INTERMEDIATE_SIZE = 458,752 neurons.
BUDGETS = (459, 2294, 4588, 9175)
DEFAULT_BUDGET = 2294


METHOD_SPECS = {
    "siren": {
        "display": "SIREN",
        "paper": "Jiao et al. — LLM Safety From Within (SIREN)",
        "family": "siren",
        "kind": "siren_json",
        "path": ("rachita_neurons/llama3-8b-instruct_mlpneuron_mean-std-"
                 "mlpneuron_mean-clean_selected_neurons_top{n}.json"),
    },
    "wang": {
        "display": "Wang",
        "paper": "Wang — Neuron-Level Safety Alignment for LLMs",
        "family": "wang",
        "kind": "csv",
        "path": "svea_neurons/fulltest_neurons_wang_N{n}.csv",
    },
    "wang_robust": {
        "display": "Wang (robust)",
        "paper": "Wang — Neuron-Level Safety Alignment for LLMs (robust variant)",
        "family": "wang",
        "kind": "csv",
        "path": "svea_neurons/fulltest_neurons_wang_robust_N{n}.csv",
    },
    "zhao_topk": {
        "display": "Zhao",
        "paper": "Zhao — Understanding and Enhancing Safety Mechanisms (top-k)",
        "family": "zhao",
        "kind": "csv",
        "path": "svea_neurons/fulltest_neurons_zhao_topk_N{n}.csv",
    },
    "zhao_eps": {
        "display": "Zhao (rel-eps)",
        "paper": "Zhao — Understanding and Enhancing Safety Mechanisms (relative epsilon)",
        "family": "zhao",
        "kind": "csv",
        "path": "svea_neurons/fulltest_neurons_zhao_relative_epsilon_N{n}.csv",
    },
    "yang_refusal": {
        "display": "Yang (refusal)",
        "paper": "Yang et al. — DPO neuron analysis (delta refusal projection)",
        "family": "yang",
        "kind": "csv",
        "path": "tengerleg_neurons/neurons_delta_refusal_N{n}.csv",
    },
    "yang_harmfulness": {
        "display": "Yang (harmfulness)",
        "paper": "Yang et al. — DPO neuron analysis (delta harmfulness projection)",
        "family": "yang",
        "kind": "csv",
        "path": "tengerleg_neurons/neurons_delta_harmfulness_N{n}.csv",
    },
}

# One canonical variant per method: the 4x4 headline comparison. Yang's RMS
# ranking no longer exists in the updated exports; refusal is the stand-in
# (swap for yang_harmfulness via --methods if preferred).
DEFAULT_METHODS = ("siren", "wang", "zhao_topk", "yang_refusal")
ALL_METHODS = tuple(METHOD_SPECS)


def display_name(method):
    return METHOD_SPECS[method]["display"]


def selection_path(method, budget):
    spec = METHOD_SPECS[method]
    return os.path.join(RESULTS_DIR, spec["path"].format(n=budget))


def load_selection(method, budget=DEFAULT_BUDGET, results_dir=None):
    """dict[layer] -> np.ndarray of neuron indices, for one method/budget."""
    if method not in METHOD_SPECS:
        raise KeyError(f"unknown method '{method}'; known: {sorted(METHOD_SPECS)}")
    spec = METHOD_SPECS[method]
    base = results_dir or RESULTS_DIR
    path = os.path.join(base, spec["path"].format(n=budget))
    if not os.path.exists(path):
        raise FileNotFoundError(f"{method} @ N={budget}: missing {path}")

    if spec["kind"] == "siren_json":
        with open(path) as f:
            raw = json.load(f)
        sel = {int(k[len("layer"):]): np.asarray(v, dtype=np.int64)
               for k, v in raw.items() if v}
    else:
        df = pd.read_csv(path)
        sel = {int(layer): sub["neuron_index"].to_numpy(dtype=np.int64)
               for layer, sub in df.groupby("layer")}

    out = {}
    for layer, idx in sel.items():
        idx = np.unique(idx)
        if idx.size == 0:
            continue
        if layer < 0 or layer >= NUM_LAYERS:
            raise ValueError(f"{method}: layer {layer} outside 0..{NUM_LAYERS - 1}")
        if idx.min() < 0 or idx.max() >= INTERMEDIATE_SIZE:
            raise ValueError(f"{method} layer {layer}: neuron index outside "
                             f"0..{INTERMEDIATE_SIZE - 1}")
        out[layer] = idx
    return out


def size(sel):
    return int(sum(len(v) for v in sel.values()))


def layer_counts(sel):
    return {l: len(v) for l, v in sorted(sel.items())}


def flatten(sel):
    """(layers, indices) aligned arrays over all selected (layer, neuron) pairs."""
    layers, idx = [], []
    for layer in sorted(sel):
        layers.append(np.full(len(sel[layer]), layer, dtype=np.int64))
        idx.append(sel[layer])
    if not layers:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(layers), np.concatenate(idx)


def jaccard(sel_a, sel_b):
    """Index-level Jaccard over (layer, neuron) pairs -- the quantity that is
    near zero across methods and motivates the CKA analysis in the first
    place. Kept here so the CKA report can print it alongside."""
    a = set(zip(*(x.tolist() for x in flatten(sel_a))))
    b = set(zip(*(x.tolist() for x in flatten(sel_b))))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ------------------------------------------------------------------ controls

def random_layer_matched(sel, rng, intermediate_size=INTERMEDIATE_SIZE):
    """Random neurons with the SAME per-layer counts as `sel`.

    This is the primary null. Because all methods draw from one shared MLP
    population, two arbitrary subsets already share the model's global
    variance structure, so a raw CKA number is uninterpretable without this
    baseline. Matching layer counts additionally removes "the two methods
    simply picked similar layers" as an explanation.
    """
    return {layer: rng.choice(intermediate_size, size=len(idx), replace=False)
            for layer, idx in sel.items()}


def random_global(sel, rng, num_layers=NUM_LAYERS,
                  intermediate_size=INTERMEDIATE_SIZE):
    """Random neurons of the same TOTAL size, spread uniformly over all layers.

    Looser null: it does not preserve the layer profile, so comparing it with
    `random_layer_matched` isolates how much of any similarity is explained by
    layer placement alone.
    """
    total = size(sel)
    flat = rng.choice(num_layers * intermediate_size, size=total, replace=False)
    layers, idx = np.divmod(flat, intermediate_size)
    return {int(l): np.sort(idx[layers == l]) for l in np.unique(layers)}


def split_halves(sel, rng):
    """Split a selection into two disjoint, layer-matched halves.

    CKA(half A, half B) of ONE method is the practical ceiling: it is what
    "the same information, found by the same procedure" scores on this data.
    Cross-method CKA should be read relative to it, not relative to 1.0.
    """
    first, second = {}, {}
    for layer, idx in sel.items():
        perm = rng.permutation(len(idx))
        cut = len(idx) // 2
        if cut == 0:
            continue
        first[layer] = idx[perm[:cut]]
        second[layer] = idx[perm[cut:2 * cut]]
    return first, second


# ------------------------------------------------------------------ matrices

def build_matrix(acts, sel, dtype=np.float32):
    """[N, num_selected] activation matrix for one selection.

    acts: [N, num_layers, intermediate_size] array (np.ndarray or memmap).
    Column order is (layer asc, then the layer's neuron index order).
    """
    blocks = [np.asarray(acts[:, layer, sel[layer]], dtype=dtype)
              for layer in sorted(sel)]
    if not blocks:
        raise ValueError("empty selection")
    return np.concatenate(blocks, axis=1)


def build_layer_matrix(acts, sel, layer, dtype=np.float32):
    """[N, k_layer] matrix for one method restricted to a single layer."""
    if layer not in sel:
        return None
    return np.asarray(acts[:, layer, sel[layer]], dtype=dtype)


def describe(method, budget=DEFAULT_BUDGET):
    sel = load_selection(method, budget)
    counts = layer_counts(sel)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return (f"{display_name(method):18s} N={size(sel):6d}  "
            f"layers used={len(counts):2d}/{NUM_LAYERS}  biggest={top}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the neuron selections.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        choices=BUDGETS)
    args = parser.parse_args()

    print(f"Neuron selections at budget N={args.budget}\n")
    for m in ALL_METHODS:
        try:
            print("  " + describe(m, args.budget))
        except FileNotFoundError as exc:
            print(f"  {display_name(m):18s} MISSING ({exc})")

    print("\nPairwise index-level Jaccard (the near-zero overlap CKA is meant "
          "to look past):")
    sels = {}
    for m in DEFAULT_METHODS:
        try:
            sels[m] = load_selection(m, args.budget)
        except FileNotFoundError:
            pass
    names = list(sels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {display_name(a):14s} vs {display_name(b):14s} "
                  f"{jaccard(sels[a], sels[b]):.4f}")

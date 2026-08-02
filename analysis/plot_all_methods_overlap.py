"""
Cross-method comparison of safety-neuron selections from three pipelines:

  1. SIREN (Rachita): per-layer L1 probes on the residual stream (4096-dim),
     neurons selected by cumulative |weight| importance. Two training sets:
     Aegis 2.0 and HH-RLHF.
  2. Zhao reproduction (Svea): per-layer neuron sets per component
     (ffn 14336-dim, attn_q/k/v/o projection rows), stored as a DataFrame
     with columns [component, layer, neuron_index, selection_mode].
  3. DPO probes (Tengerleg): two single 4096-dim residual-stream logistic
     probes (refusal @ t_post-inst, harmfulness @ t_inst), layer-agnostic.

Index-level Jaccard is only meaningful between selections living in the SAME
index space. That holds for SIREN<->SIREN and SIREN<->Tengerleg (both are
residual-stream dims). Svea's indices are MLP/attention component rows, so
she is compared at the layer-distribution level instead, with her
MLP / attention / total split shown explicitly.

Usage:

    python analysis/plot_all_methods_overlap.py \
        --siren_aegis results/llama3-8b-instruct_general_probes-aegis.pkl \
        --siren_hhrlhf results/llama3-8b-instruct_general_probes-hhrlhf.pkl \
        --svea results/probes-svea.pkl \
        --tengerleg results/probes-tengerleg.pkl
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import io
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- selection

def select_cumulative(weights, threshold):
    """SIREN's rule: top neurons covering `threshold` of total |weight|."""
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(weights)[::-1]
    cum = np.cumsum(weights[order])
    n = int(np.searchsorted(cum, threshold * cum[-1])) + 1
    return set(int(i) for i in order[:n])


def select_top_k(weights, k):
    order = np.argsort(np.asarray(weights))[::-1]
    return set(int(i) for i in order[:k])


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ------------------------------------------------------------------ loaders

class CPUUnpickler(pickle.Unpickler):
    """Maps CUDA tensors to CPU (SIREN probes were saved on GPU)."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


def load_siren(path, pooling_type, threshold):
    """{layer: set(residual-stream neuron indices)} plus per-layer weights."""
    with open(path, "rb") as f:
        data = CPUUnpickler(f).load()
    neurons, weights = {}, {}
    suffix = f"_{pooling_type}"
    for key, entry in data["best_probes"].items():
        if not key.endswith(suffix):
            continue
        layer = int(key[len("layer"):-len(suffix)])
        w = entry["probe"].get_feature_importance()
        weights[layer] = w
        neurons[layer] = select_cumulative(w, threshold)
    return neurons, weights


def load_svea(path):
    """DataFrame -> per-layer counts for mlp / attn / total.

    Returns a DataFrame indexed by layer with columns [mlp, attn, total],
    plus the raw per-layer neuron sets per component (for completeness).
    """
    with open(path, "rb") as f:
        df = pickle.load(f)
    df = df.copy()
    df["group"] = np.where(df["component"] == "ffn", "mlp", "attn")
    counts = df.groupby(["layer", "group"]).size().unstack(fill_value=0)
    for col in ("mlp", "attn"):
        if col not in counts:
            counts[col] = 0
    counts["total"] = counts["mlp"] + counts["attn"]
    sets = {
        group: {layer: set(sub["neuron_index"].astype(int))
                for layer, sub in g.groupby("layer")}
        for group, g in df.groupby("group")
    }
    return counts.sort_index(), sets


def load_tengerleg(path):
    """{probe_name: abs weight vector (4096,)}"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "refusal": np.abs(data["refusal_probe"].coef_[0]),
        "harmfulness": np.abs(data["harmfulness_probe"].coef_[0]),
    }


# -------------------------------------------------------------------- plots

def plot_counts(siren_counts, svea_counts, teng_sizes, output_dir,
                teng_totals=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for label, counts in siren_counts.items():
        layers = sorted(counts)
        ax.plot(layers, [counts[l] for l in layers], marker="o",
                markersize=3.5, label=label)
    ax.plot(svea_counts.index, svea_counts["mlp"], marker="^", markersize=3.5,
            linestyle="--", label="svea/zhao (MLP)")
    ax.plot(svea_counts.index, svea_counts["attn"], marker="v", markersize=3.5,
            linestyle="--", label="svea/zhao (attention)")
    ax.plot(svea_counts.index, svea_counts["total"], marker="D", markersize=3.5,
            label="svea/zhao (total)")
    if teng_totals:
        # Tengerleg's probes are layer-agnostic (one 4096-dim weight vector
        # for the whole model), so per layer they can only appear as a flat
        # reference line at the probe's total selection size.
        for (name, total), color in zip(teng_totals.items(),
                                        ("tab:red", "tab:cyan")):
            ax.axhline(total, color=color, linestyle=":", linewidth=1.5,
                       label=f"tengerleg-{name} (layer-agnostic total)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of selected safety neurons")
    ax.set_title("Selected safety neurons per layer (llama3-8b-instruct)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = list(teng_sizes.keys())
    values = [teng_sizes[k] for k in labels]
    bars = ax.bar(range(len(labels)), values, color="tab:gray")
    ax.bar_label(bars, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total selected neurons")
    ax.set_title("Layer-agnostic totals\n(Tengerleg probes have no layer axis)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "all_methods_neuron_counts.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_layer_profiles(siren_counts, svea_counts, output_dir):
    """Share of each method's selected neurons per layer -- the cross-method
    comparison that stays valid even when neuron index spaces differ."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, counts in siren_counts.items():
        layers = sorted(counts)
        vals = np.array([counts[l] for l in layers], dtype=float)
        ax.plot(layers, vals / vals.sum(), marker="o", markersize=3.5, label=label)
    for group, style in (("mlp", "--"), ("attn", "--"), ("total", "-")):
        vals = svea_counts[group].to_numpy(dtype=float)
        ax.plot(svea_counts.index, vals / vals.sum(), marker="^", markersize=3.5,
                linestyle=style, label=f"svea/zhao ({group})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of method's selected neurons")
    ax.set_title("Layer distribution of selected safety neurons (normalized per method)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "all_methods_layer_profiles.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_siren_heatmap(neurons_a, neurons_b, label_a, label_b, threshold, output_dir):
    layers_a, layers_b = sorted(neurons_a), sorted(neurons_b)
    matrix = np.array([[jaccard(neurons_a[la], neurons_b[lb]) for la in layers_a]
                       for lb in layers_b])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, len(layers_a), 4))
    ax.set_xticklabels(layers_a[::4])
    ax.set_yticks(range(0, len(layers_b), 4))
    ax.set_yticklabels(layers_b[::4])
    ax.set_xlabel(f"{label_a} layer")
    ax.set_ylabel(f"{label_b} layer")
    ax.set_title(f"Jaccard of selected neurons (threshold={threshold})")
    fig.colorbar(im, ax=ax, label="Jaccard")

    common = sorted(set(layers_a) & set(layers_b))
    diag = [jaccard(neurons_a[l], neurons_b[l]) for l in common]
    ax = axes[1]
    ax.plot(common, diag, marker="o", markersize=4, color="tab:purple",
            label="observed")
    # Chance level for two random subsets of sizes k_a, k_b in 4096 dims:
    # E[intersection] = k_a*k_b/4096, E[J] = inter / (k_a + k_b - inter).
    chance = []
    for l in common:
        ka, kb = len(neurons_a[l]), len(neurons_b[l])
        inter = ka * kb / 4096
        chance.append(inter / (ka + kb - inter))
    ax.plot(common, chance, color="gray", linestyle=":", label="chance level")
    ax.legend(fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Same-layer overlap: {label_a} vs {label_b}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"all_methods_jaccard_{label_a}_vs_{label_b}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_svea_vs_siren(svea_sets, siren_neurons, siren_label, output_dir):
    """Raw index-level Jaccard between Svea's total per-layer selection
    (union of ffn + attention neuron indices, taken as plain integers) and a
    SIREN run's per-layer residual-stream sets.

    CAVEAT: the index spaces differ. Svea's ffn indices (0..14335) index MLP
    rows and can never coincide with residual dims; her attention indices
    (0..4095) index projection rows that merely share the integer range with
    residual dims. Matches are therefore integer collisions, not "the same
    neuron". The chance line models exactly that: only her in-range
    (< 4096) indices can collide, and collisions are random.
    """
    svea_total = {}
    for layer in set().union(*(set(s) for s in svea_sets.values())):
        svea_total[layer] = set()
        for comp_sets in svea_sets.values():
            svea_total[layer] |= comp_sets.get(layer, set())

    layers_svea = sorted(svea_total)
    layers_siren = sorted(siren_neurons)
    matrix = np.array([[jaccard(siren_neurons[ls], svea_total[lv])
                        for ls in layers_siren] for lv in layers_svea])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, len(layers_siren), 4))
    ax.set_xticklabels(layers_siren[::4])
    ax.set_yticks(range(0, len(layers_svea), 4))
    ax.set_yticklabels(layers_svea[::4])
    ax.set_xlabel(f"{siren_label} layer")
    ax.set_ylabel("svea/zhao layer")
    ax.set_title("Jaccard of neuron indices (CAVEAT: different index spaces)")
    fig.colorbar(im, ax=ax, label="Jaccard")

    common = sorted(set(layers_svea) & set(layers_siren))
    diag, chance = [], []
    for l in common:
        s_siren, s_svea = siren_neurons[l], svea_total[l]
        diag.append(jaccard(s_siren, s_svea))
        k_in = sum(1 for i in s_svea if i < 4096)
        inter = len(s_siren) * k_in / 4096
        chance.append(inter / (len(s_siren) + len(s_svea) - inter))
    ax = axes[1]
    ax.plot(common, diag, marker="o", markersize=4, color="tab:green",
            label="observed")
    ax.plot(common, chance, color="gray", linestyle=":",
            label="chance level (integer collisions)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Same-layer: svea/zhao (total) vs {siren_label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"all_methods_jaccard_svea_vs_{siren_label}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_tengerleg_vs_siren(siren_neurons, teng_weights, threshold, output_dir):
    """Tengerleg's probes are layer-agnostic, so the comparison is one line
    per (SIREN run, Tengerleg probe): Jaccard of the Tengerleg set against
    each SIREN layer's set. To avoid the set-size artifact (her dense L2
    probe selects far more neurons under the cumulative rule than SIREN's
    sparse L1 probes), the Tengerleg set is size-matched per layer: her
    top-k neurons by |weight|, k = |SIREN layer set|."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"refusal": "tab:red", "harmfulness": "tab:blue"}
    for si, (siren_label, neurons) in enumerate(siren_neurons.items()):
        style = "-" if si == 0 else "--"
        layers = sorted(neurons)
        for probe_name, w in teng_weights.items():
            vals = [jaccard(neurons[l], select_top_k(w, len(neurons[l])))
                    for l in layers]
            ax.plot(layers, vals, marker="o", markersize=3,
                    color=colors[probe_name], linestyle=style,
                    label=f"{siren_label} vs tengerleg-{probe_name}")
        # Chance level for two random size-k subsets of a 4096-dim space:
        # E[Jaccard] approx k / (2*4096 - k). Depends on each run's set
        # sizes, so plot one baseline per SIREN run.
        ks = np.array([len(neurons[l]) for l in layers], dtype=float)
        ax.plot(layers, ks / (2 * 4096 - ks), color="gray", linestyle=style,
                linewidth=1, alpha=0.8,
                label=f"chance level ({siren_label} set sizes)")

    teng_sets = {n: select_cumulative(w, threshold) for n, w in teng_weights.items()}
    j_rh = jaccard(teng_sets["refusal"], teng_sets["harmfulness"])
    ax.set_xlabel("SIREN layer")
    ax.set_ylabel("Jaccard similarity (size-matched top-k)")
    ax.set_title("Residual-stream neuron overlap: SIREN layers vs Tengerleg probes\n"
                 f"(tengerleg refusal vs harmfulness Jaccard @ t={threshold}: {j_rh:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "all_methods_jaccard_tengerleg_vs_siren.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
    return j_rh


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siren_aegis", required=True)
    parser.add_argument("--siren_hhrlhf", required=True)
    parser.add_argument("--svea", required=True)
    parser.add_argument("--tengerleg", required=True)
    parser.add_argument("--pooling_type", default="residual_mean")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    siren = {}
    for label, path in (("siren-aegis", args.siren_aegis),
                        ("siren-hhrlhf", args.siren_hhrlhf)):
        neurons, _ = load_siren(path, args.pooling_type, args.threshold)
        siren[label] = neurons
    svea_counts, _svea_sets = load_svea(args.svea)
    teng_weights = load_tengerleg(args.tengerleg)
    teng_sets = {f"tengerleg-{n}": select_cumulative(w, args.threshold)
                 for n, w in teng_weights.items()}

    siren_counts = {label: {l: len(s) for l, s in neurons.items()}
                    for label, neurons in siren.items()}
    totals = {label: sum(c.values()) for label, c in siren_counts.items()}
    totals["svea/zhao (total)"] = int(svea_counts["total"].sum())
    totals.update({n: len(s) for n, s in teng_sets.items()})

    teng_totals = {n: len(select_cumulative(w, args.threshold))
                   for n, w in teng_weights.items()}
    plot_counts(siren_counts, svea_counts, totals, args.output_dir,
                teng_totals=teng_totals)
    plot_layer_profiles(siren_counts, svea_counts, args.output_dir)
    plot_siren_heatmap(siren["siren-aegis"], siren["siren-hhrlhf"],
                       "siren-aegis", "siren-hhrlhf", args.threshold,
                       args.output_dir)
    for label, neurons in siren.items():
        plot_svea_vs_siren(_svea_sets, neurons, label, args.output_dir)
    j_rh = plot_tengerleg_vs_siren(siren, teng_weights, args.threshold,
                                   args.output_dir)

    print("\nTotal selected neurons per method:")
    for k, v in totals.items():
        print(f"  {k:28s} {v}")
    print(f"\nTengerleg refusal vs harmfulness Jaccard (t={args.threshold}): {j_rh:.4f}")
    print("\nNote: Svea's neuron indices live in MLP (14336-dim) and attention "
          "projection-row spaces, not the residual stream, so index-level "
          "Jaccard against SIREN/Tengerleg is not defined; she is compared "
          "via per-layer counts and normalized layer profiles instead.")


if __name__ == "__main__":
    main()

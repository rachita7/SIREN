"""
Layer-wise overlap statistics between safety-neuron identification methods.

Methods compared:
  - SIREN (Rachita): LLM Safety From Within
  - Zhao top-k: Zhao (Understanding and Enhancing Safety Mechanism)
  - Zhao relative epsilon: Zhao (variant)
  - Wang standard: Wang (Neuron-Level Safety Alignment for LLMs)
  - Wang robust: Wang (variant)
  - Tengerleg RMS: Yang et al. (How Does DPO Reduce Toxicity?) - RMS metric
  - Tengerleg delta-refusal: Yang et al. - delta refusal metric
  - Tengerleg delta-harmfulness: Yang et al. - delta harmfulness metric

All methods identify neurons in the MLP/FFN space (14336 neurons × 32 layers).
Layer-wise Jaccard similarity is directly meaningful.

Produces for each neuron count (2500, 5000, 10000):
  1. Layer distribution: neurons per layer for all methods
  2. Per-layer pairwise Jaccard line plots (grouped by method family)
  3. Summary heatmap: mean Jaccard across layers for all method pairs

Usage:
    python analysis/plot_layerwise_overlap.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
NUM_LAYERS = 32
MLP_DIM = 14336

METHOD_LABELS = {
    "siren": "SIREN",
    "zhao_topk": "Zhao (top-k)",
    "zhao_relative_epsilon": "Zhao (rel. epsilon)",
    "wang": "Wang (standard)",
    "wang_robust": "Wang (robust)",
    "rms": "Yang et al. (RMS)",
    "delta_refusal": "Yang et al. (delta-refusal)",
    "delta_harmfulness": "Yang et al. (delta-harmful.)",
}

METHOD_COLORS = {
    "siren": "#E63946",
    "zhao_topk": "#457B9D",
    "zhao_relative_epsilon": "#1D3557",
    "wang": "#2A9D8F",
    "wang_robust": "#264653",
    "rms": "#E9C46A",
    "delta_refusal": "#F4A261",
    "delta_harmfulness": "#E76F51",
}

METHOD_STYLES = {
    "siren": "-",
    "zhao_topk": "--",
    "zhao_relative_epsilon": "-.",
    "wang": "--",
    "wang_robust": "-.",
    "rms": "--",
    "delta_refusal": "-.",
    "delta_harmfulness": ":",
}

NEURON_COUNTS = [2500, 5000, 10000]


def load_siren(n):
    """Load SIREN neurons from JSON -> {layer_int: set(neuron_indices)}."""
    path = os.path.join(RESULTS_DIR, "rachita_neurons", f"siren_top{n}.json")
    with open(path) as f:
        data = json.load(f)
    return {int(k.replace("layer", "")): set(v) for k, v in data.items()}


def load_svea(variant, n):
    """Load Zhao/Wang neurons from CSV -> {layer_int: set(neuron_indices)}."""
    path = os.path.join(
        RESULTS_DIR, "svea_neurons",
        f"intersection_neurons_{variant}_N{n}_intersectiontuned.csv"
    )
    df = pd.read_csv(path)
    result = {}
    for layer, group in df.groupby("layer"):
        result[int(layer)] = set(group["neuron_index"].astype(int))
    return result


def load_tengerleg(metric, n):
    """Load Tengerleg neurons from CSV -> {layer_int: set(neuron_indices)}."""
    path = os.path.join(
        RESULTS_DIR, "tengerleg_neurons",
        f"{metric}_ranking_top{n}.csv"
    )
    df = pd.read_csv(path)
    result = {}
    for layer, group in df.groupby("layer"):
        result[int(layer)] = set(group["neuron_idx"].astype(int))
    return result


def load_all_methods(n):
    """Load all 8 methods for a given neuron count n."""
    methods = {}
    methods["siren"] = load_siren(n)
    for variant in ["zhao_topk", "zhao_relative_epsilon"]:
        methods[variant] = load_svea(variant, n)
    for variant in ["wang", "wang_robust"]:
        methods[variant] = load_svea(variant, n)
    for metric in ["rms", "delta_refusal", "delta_harmfulness"]:
        methods[metric] = load_tengerleg(metric, n)
    return methods


def jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def layer_distribution(methods):
    """Return DataFrame: rows=layers, columns=methods, values=neuron count."""
    data = {}
    for name, layer_sets in methods.items():
        counts = [len(layer_sets.get(l, set())) for l in range(NUM_LAYERS)]
        data[name] = counts
    return pd.DataFrame(data, index=range(NUM_LAYERS))


def pairwise_jaccard_per_layer(methods):
    """Compute Jaccard per layer for all method pairs.

    Returns dict: (method_a, method_b) -> array of shape (NUM_LAYERS,)
    """
    method_names = list(methods.keys())
    result = {}
    for i, a in enumerate(method_names):
        for j, b in enumerate(method_names):
            if j <= i:
                continue
            jaccards = []
            for layer in range(NUM_LAYERS):
                set_a = methods[a].get(layer, set())
                set_b = methods[b].get(layer, set())
                jaccards.append(jaccard(set_a, set_b))
            result[(a, b)] = np.array(jaccards)
    return result


def chance_jaccard_per_layer(methods_a_name, methods_b_name, methods):
    """Expected Jaccard for two random subsets of size k_a, k_b from MLP_DIM."""
    chance = []
    for layer in range(NUM_LAYERS):
        ka = len(methods[methods_a_name].get(layer, set()))
        kb = len(methods[methods_b_name].get(layer, set()))
        if ka == 0 or kb == 0:
            chance.append(0.0)
        else:
            expected_inter = ka * kb / MLP_DIM
            chance.append(expected_inter / (ka + kb - expected_inter))
    return np.array(chance)


# ========== PLOT 1: Layer distribution ==========

def plot_layer_distribution(methods, n, output_dir):
    """Line plot: neurons per layer for all methods."""
    dist = layer_distribution(methods)

    fig, ax = plt.subplots(figsize=(12, 6))
    for method in methods:
        ax.plot(
            range(NUM_LAYERS), dist[method],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=2, marker="o", markersize=3
        )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Number of neurons", fontsize=12)
    ax.set_title(f"Layer distribution of top-{n} safety neurons", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"layer_distribution_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ========== PLOT 2: Pairwise Jaccard summary heatmap ==========

def plot_jaccard_heatmap(methods, n, output_dir):
    """Heatmap of mean Jaccard (across all layers) for each method pair."""
    method_names = list(methods.keys())
    num_methods = len(method_names)
    pairwise = pairwise_jaccard_per_layer(methods)

    matrix = np.zeros((num_methods, num_methods))
    np.fill_diagonal(matrix, 1.0)
    for (a, b), jac_arr in pairwise.items():
        i, j = method_names.index(a), method_names.index(b)
        mean_j = jac_arr.mean()
        matrix[i, j] = mean_j
        matrix[j, i] = mean_j

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(0.3, matrix[matrix < 1.0].max() * 1.2))
    labels = [METHOD_LABELS[m] for m in method_names]
    ax.set_xticks(range(num_methods))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(num_methods))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(num_methods):
        for j in range(num_methods):
            val = matrix[i, j]
            color = "white" if val > 0.15 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(f"Mean layer-wise Jaccard similarity (top-{n} neurons)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Mean Jaccard", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"jaccard_heatmap_all_methods_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")

    csv_path = os.path.join(output_dir, f"jaccard_mean_matrix_top{n}.csv")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path)
    print(f"  Saved: {csv_path}")


# ========== PLOT 3: Per-layer Jaccard line plots (SIREN vs all others) ==========

def plot_siren_vs_others(methods, n, output_dir):
    """Line plot: per-layer Jaccard of SIREN vs each other method."""
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(NUM_LAYERS)

    for other in list(methods.keys()):
        if other == "siren":
            continue
        jac = []
        for l in range(NUM_LAYERS):
            set_siren = methods["siren"].get(l, set())
            set_other = methods[other].get(l, set())
            jac.append(jaccard(set_siren, set_other))
        ax.plot(
            layers, jac,
            label=METHOD_LABELS[other],
            color=METHOD_COLORS[other],
            linestyle=METHOD_STYLES[other],
            linewidth=2, marker="o", markersize=3
        )

    chance = chance_jaccard_per_layer("siren", "zhao_topk", methods)
    ax.plot(layers, chance, color="gray", linestyle=":", linewidth=1.5,
            alpha=0.7, label="Chance level")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Jaccard similarity", fontsize=12)
    ax.set_title(f"SIREN vs other methods: per-layer Jaccard (top-{n})", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"siren_vs_others_jaccard_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ========== PLOT 4: Within-family comparisons ==========

def plot_within_family(methods, n, output_dir):
    """Per-layer Jaccard within each method family (Zhao, Wang, Tengerleg)."""
    families = {
        "Zhao": ("zhao_topk", "zhao_relative_epsilon"),
        "Wang": ("wang", "wang_robust"),
        "Yang et al.": [("rms", "delta_refusal"), ("rms", "delta_harmfulness"),
                        ("delta_refusal", "delta_harmfulness")],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Zhao
    ax = axes[0]
    a, b = families["Zhao"]
    jac = [jaccard(methods[a].get(l, set()), methods[b].get(l, set()))
           for l in range(NUM_LAYERS)]
    ax.plot(range(NUM_LAYERS), jac, linewidth=2, marker="o", markersize=4,
            color="#457B9D", label=f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}")
    chance = chance_jaccard_per_layer(a, b, methods)
    ax.plot(range(NUM_LAYERS), chance, color="gray", linestyle=":", linewidth=1.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Jaccard similarity", fontsize=11)
    ax.set_title(f"Zhao: within-family overlap (top-{n})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.grid(alpha=0.3)

    # Wang
    ax = axes[1]
    a, b = families["Wang"]
    jac = [jaccard(methods[a].get(l, set()), methods[b].get(l, set()))
           for l in range(NUM_LAYERS)]
    ax.plot(range(NUM_LAYERS), jac, linewidth=2, marker="o", markersize=4,
            color="#2A9D8F", label=f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}")
    chance = chance_jaccard_per_layer(a, b, methods)
    ax.plot(range(NUM_LAYERS), chance, color="gray", linestyle=":", linewidth=1.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Jaccard similarity", fontsize=11)
    ax.set_title(f"Wang: within-family overlap (top-{n})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.grid(alpha=0.3)

    # Tengerleg (3 pairs)
    ax = axes[2]
    pair_colors = ["#E9C46A", "#F4A261", "#E76F51"]
    for idx, (a, b) in enumerate(families["Yang et al."]):
        jac = [jaccard(methods[a].get(l, set()), methods[b].get(l, set()))
               for l in range(NUM_LAYERS)]
        ax.plot(range(NUM_LAYERS), jac, linewidth=2, marker="o", markersize=3,
                color=pair_colors[idx],
                label=f"{METHOD_LABELS[a]} vs\n{METHOD_LABELS[b]}")
    chance = chance_jaccard_per_layer("rms", "delta_refusal", methods)
    ax.plot(range(NUM_LAYERS), chance, color="gray", linestyle=":", linewidth=1.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Jaccard similarity", fontsize=11)
    ax.set_title(f"Yang et al.: within-family overlap (top-{n})", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"within_family_jaccard_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ========== PLOT 5: Cross-family per-layer Jaccard (grouped) ==========

def plot_cross_family(methods, n, output_dir):
    """Per-layer Jaccard between method families (Zhao vs Wang, Zhao vs Yang, etc.)."""
    cross_pairs = [
        ("zhao_topk", "wang", "Zhao (top-k) vs Wang"),
        ("zhao_topk", "rms", "Zhao (top-k) vs Yang (RMS)"),
        ("wang", "rms", "Wang vs Yang (RMS)"),
        ("zhao_relative_epsilon", "wang_robust", "Zhao (rel.eps) vs Wang (robust)"),
        ("zhao_topk", "delta_refusal", "Zhao (top-k) vs Yang (delta-ref.)"),
        ("wang", "delta_harmfulness", "Wang vs Yang (delta-harm.)"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cross_pairs)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (a, b, label) in enumerate(cross_pairs):
        jac = [jaccard(methods[a].get(l, set()), methods[b].get(l, set()))
               for l in range(NUM_LAYERS)]
        ax.plot(range(NUM_LAYERS), jac, linewidth=1.8, marker="o", markersize=3,
                color=colors[idx], label=label)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Jaccard similarity", fontsize=12)
    ax.set_title(f"Cross-family per-layer Jaccard (top-{n})", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"cross_family_jaccard_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ========== PLOT 6: Normalized layer profiles (fraction of neurons per layer) ==========

def plot_normalized_profiles(methods, n, output_dir):
    """Fraction of each method's total neurons that fall in each layer."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in methods:
        counts = np.array([len(methods[method].get(l, set())) for l in range(NUM_LAYERS)],
                          dtype=float)
        total = counts.sum()
        if total == 0:
            continue
        fractions = counts / total
        ax.plot(
            range(NUM_LAYERS), fractions,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=2, marker="o", markersize=3
        )

    uniform = np.ones(NUM_LAYERS) / NUM_LAYERS
    ax.axhline(uniform[0], color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Uniform")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Fraction of total neurons", fontsize=12)
    ax.set_title(f"Normalized layer profiles (top-{n} neurons)", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"normalized_layer_profiles_top{n}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# ========== Summary statistics ==========

def print_summary(methods, pairwise, n):
    """Print summary statistics for the given neuron count."""
    print(f"\n{'='*60}")
    print(f"  SUMMARY STATISTICS: top-{n} neurons")
    print(f"{'='*60}")

    print(f"\n  Layer distribution (neurons per layer, mean ± std):")
    for method in methods:
        counts = [len(methods[method].get(l, set())) for l in range(NUM_LAYERS)]
        total = sum(counts)
        print(f"    {METHOD_LABELS[method]:30s}: total={total:5d}, "
              f"mean={np.mean(counts):.1f} ± {np.std(counts):.1f}, "
              f"max={max(counts)} (L{np.argmax(counts)}), "
              f"min={min(counts)} (L{np.argmin(counts)})")

    print(f"\n  Top-5 method pairs by mean Jaccard (layer-wise):")
    ranked = sorted(pairwise.items(), key=lambda x: x[1].mean(), reverse=True)
    for (a, b), jac_arr in ranked[:5]:
        print(f"    {METHOD_LABELS[a]:30s} vs {METHOD_LABELS[b]:30s}: "
              f"mean={jac_arr.mean():.4f}, max={jac_arr.max():.4f} (L{jac_arr.argmax()})")

    print(f"\n  Bottom-5 method pairs by mean Jaccard (layer-wise):")
    for (a, b), jac_arr in ranked[-5:]:
        print(f"    {METHOD_LABELS[a]:30s} vs {METHOD_LABELS[b]:30s}: "
              f"mean={jac_arr.mean():.4f}, max={jac_arr.max():.4f} (L{jac_arr.argmax()})")


# ========== Main ==========

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Layer-wise overlap analysis across safety-neuron methods")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for n in NEURON_COUNTS:
        print(f"\n{'#'*60}")
        print(f"  Processing top-{n} neurons")
        print(f"{'#'*60}")

        methods = load_all_methods(n)
        pairwise = pairwise_jaccard_per_layer(methods)

        plot_layer_distribution(methods, n, OUTPUT_DIR)
        plot_normalized_profiles(methods, n, OUTPUT_DIR)
        plot_jaccard_heatmap(methods, n, OUTPUT_DIR)
        plot_siren_vs_others(methods, n, OUTPUT_DIR)
        plot_within_family(methods, n, OUTPUT_DIR)
        plot_cross_family(methods, n, OUTPUT_DIR)
        print_summary(methods, pairwise, n)

    print(f"\n\nDone! All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

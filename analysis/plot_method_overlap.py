"""
Compare safety-neuron selections of two SIREN training methods (e.g. the
Aegis-trained probes vs. the HH-RLHF-trained probes on the same backbone).

Produces:
  1. A line plot with the number of selected safety neurons per layer for both
     methods (read directly from the layer-stats CSVs in results/).
  2. A cross-layer Jaccard-similarity heatmap: layers of method A on the
     X axis, layers of method B on the Y axis, cell (i, j) = Jaccard overlap
     between the neuron sets selected at layer i (A) and layer j (B).
     The diagonal (same layer in both methods) is also plotted as a line.

The neuron *indices* are not stored in the CSVs, only in the probe pickles, so
the heatmap requires the two `*_general_probes.pkl` files. The pickles contain
CUDA tensors, so run this part on a GPU machine (same constraint as
plot_layer_probes.py). The CSV-based count plot works anywhere.

Usage (counts only, runs locally):

    python analysis/plot_method_overlap.py \
        --csv_a results/llama3-8b-instruct_residual_mean_layer_stats-aegis.csv \
        --csv_b results/llama3-8b-instruct_residual_mean_layer_stats-hhrlhf.csv \
        --label_a aegis --label_b hh-rlhf

Usage (counts + Jaccard heatmap, needs the probe pickles / GPU):

    python analysis/plot_method_overlap.py \
        --csv_a results/..._layer_stats-aegis.csv \
        --csv_b results/..._layer_stats-hhrlhf.csv \
        --label_a aegis --label_b hh-rlhf \
        --probes_a train/probes/llama3-8b-instruct_general_probes-aegis.pkl \
        --probes_b train/probes/llama3-8b-instruct_general_probes-hhrlhf.pkl \
        --threshold 0.9
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def select_salient_neurons(probe, threshold):
    weights = probe.get_feature_importance()
    total_importance = np.sum(weights)
    sorted_indices = np.argsort(weights)[::-1]
    selected = []
    cumulative = 0.0
    for idx in sorted_indices:
        selected.append(int(idx))
        cumulative += weights[idx]
        if cumulative >= threshold * total_importance:
            break
    return selected


def load_selected_neurons(probe_path, pooling_type, threshold):
    """Return {layer_idx: set(neuron indices)} from a probes pickle."""
    with open(probe_path, "rb") as f:
        data = pickle.load(f)
    best_probes = data["best_probes"]

    neurons = {}
    suffix = f"_{pooling_type}"
    for key, entry in best_probes.items():
        if not key.endswith(suffix):
            continue
        layer_idx = int(key[len("layer"):-len(suffix)])
        neurons[layer_idx] = set(select_salient_neurons(entry["probe"], threshold))
    return neurons


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def plot_neuron_counts(df_a, df_b, label_a, label_b, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_a["layer"], df_a["num_selected"], marker="o", markersize=4, label=label_a)
    ax.plot(df_b["layer"], df_b["num_selected"], marker="s", markersize=4, label=label_b)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of selected safety neurons")
    model = df_a["model"].iloc[0]
    ax.set_title(f"Selected safety neurons per layer ({model})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"neuron_counts_{label_a}_vs_{label_b}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_jaccard_heatmap(neurons_a, neurons_b, label_a, label_b, threshold, output_dir):
    layers_a = sorted(neurons_a.keys())
    layers_b = sorted(neurons_b.keys())
    matrix = np.zeros((len(layers_b), len(layers_a)))
    for yi, lb in enumerate(layers_b):
        for xi, la in enumerate(layers_a):
            matrix[yi, xi] = jaccard(neurons_a[la], neurons_b[lb])

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, len(layers_a), 2))
    ax.set_xticklabels(layers_a[::2])
    ax.set_yticks(range(0, len(layers_b), 2))
    ax.set_yticklabels(layers_b[::2])
    ax.set_xlabel(f"{label_a} layer")
    ax.set_ylabel(f"{label_b} layer")
    ax.set_title(f"Jaccard similarity of selected neurons (threshold={threshold})")
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    plt.tight_layout()
    path = os.path.join(output_dir, f"jaccard_heatmap_{label_a}_vs_{label_b}_t{threshold}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    # Same-layer (diagonal) overlap as a line plot for easy reading.
    common_layers = sorted(set(layers_a) & set(layers_b))
    diag = [jaccard(neurons_a[l], neurons_b[l]) for l in common_layers]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(common_layers, diag, marker="o", markersize=4, color="tab:purple")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Same-layer neuron overlap: {label_a} vs {label_b} (threshold={threshold})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"jaccard_diagonal_{label_a}_vs_{label_b}_t{threshold}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    # CSV dump of the full matrix for further analysis.
    mat_df = pd.DataFrame(matrix,
                          index=[f"{label_b}_layer{l}" for l in layers_b],
                          columns=[f"{label_a}_layer{l}" for l in layers_a])
    csv_path = os.path.join(output_dir, f"jaccard_matrix_{label_a}_vs_{label_b}_t{threshold}.csv")
    mat_df.to_csv(csv_path)
    print(f"Saved {csv_path}")

    print(f"\nSame-layer Jaccard: mean={np.mean(diag):.4f}, "
          f"max={np.max(diag):.4f} (layer {common_layers[int(np.argmax(diag))]}), "
          f"min={np.min(diag):.4f} (layer {common_layers[int(np.argmin(diag))]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_a", type=str, required=True, help="Layer-stats CSV for method A")
    parser.add_argument("--csv_b", type=str, required=True, help="Layer-stats CSV for method B")
    parser.add_argument("--label_a", type=str, default="method_a")
    parser.add_argument("--label_b", type=str, default="method_b")
    parser.add_argument("--probes_a", type=str, default=None,
                        help="Probes pickle for method A (enables the Jaccard heatmap)")
    parser.add_argument("--probes_b", type=str, default=None,
                        help="Probes pickle for method B")
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Cumulative-importance threshold used to select neurons")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    plot_neuron_counts(df_a, df_b, args.label_a, args.label_b, args.output_dir)

    if args.probes_a and args.probes_b:
        neurons_a = load_selected_neurons(args.probes_a, args.pooling_type, args.threshold)
        neurons_b = load_selected_neurons(args.probes_b, args.pooling_type, args.threshold)
        plot_jaccard_heatmap(neurons_a, neurons_b, args.label_a, args.label_b,
                             args.threshold, args.output_dir)
    else:
        print("\nNo probe pickles given (--probes_a/--probes_b), skipping the Jaccard "
              "heatmap: the neuron indices are only stored in the pickles, not the CSVs.")


if __name__ == "__main__":
    main()

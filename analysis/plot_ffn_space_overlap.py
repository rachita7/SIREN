"""
FFN-space (14336-dim, down_proj-input) overlap between:

  1. SIREN trained with mlpneuron_mean pooling (per-layer L1 probes on the
     MLP intermediate activations, i.e. the input to mlp.down_proj).
  2. Zhao/Svea's `ffn` component selection (results/probes-svea.pkl,
     filtered to component == "ffn"), which indexes the SAME space.

Unlike the residual_mean comparison, an index-level Jaccard here is
meaningful: "neuron k at layer L" refers to the same physical MLP unit in
both methods.

Usage (after the mlpneuron_mean training run):

    python analysis/plot_ffn_space_overlap.py \
        --siren_probes results/llama3-8b-instruct_general_probes-hhrlhf-mlpneuron.pkl \
        --svea results/probes-svea.pkl \
        --threshold 0.9
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_all_methods_overlap import select_cumulative, jaccard, load_siren

FFN_DIM = 14336  # Llama-3-8B intermediate_size


def load_zhao_ffn(path):
    """{layer: set(ffn neuron indices)} from the Svea DataFrame."""
    with open(path, "rb") as f:
        df = pickle.load(f)
    ffn = df[df["component"] == "ffn"]
    return {int(layer): set(sub["neuron_index"].astype(int))
            for layer, sub in ffn.groupby("layer")}


def chance_jaccard(ka, kb, dim=FFN_DIM):
    """Expected Jaccard of two independent random subsets of sizes ka, kb."""
    inter = ka * kb / dim
    denom = ka + kb - inter
    return inter / denom if denom else 0.0


def plot_counts(siren_neurons, zhao_neurons, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    layers = sorted(siren_neurons)
    ax.plot(layers, [len(siren_neurons[l]) for l in layers], marker="o",
            markersize=3.5, label="SIREN (mlpneuron_mean)")
    layers_z = sorted(zhao_neurons)
    ax.plot(layers_z, [len(zhao_neurons[l]) for l in layers_z], marker="^",
            markersize=3.5, linestyle="--", label="zhao/svea (ffn)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of selected FFN neurons")
    ax.set_title("Selected FFN neurons per layer (14336-dim down_proj-input space)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ffn_space_neuron_counts.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_jaccard(siren_neurons, zhao_neurons, threshold, output_dir):
    layers_s = sorted(siren_neurons)
    layers_z = sorted(zhao_neurons)
    matrix = np.array([[jaccard(siren_neurons[ls], zhao_neurons[lz])
                        for ls in layers_s] for lz in layers_z])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, len(layers_s), 4))
    ax.set_xticklabels(layers_s[::4])
    ax.set_yticks(range(0, len(layers_z), 4))
    ax.set_yticklabels(layers_z[::4])
    ax.set_xlabel("SIREN (mlpneuron_mean) layer")
    ax.set_ylabel("zhao/svea (ffn) layer")
    ax.set_title(f"Jaccard of FFN neuron indices (threshold={threshold})")
    fig.colorbar(im, ax=ax, label="Jaccard")

    common = sorted(set(layers_s) & set(layers_z))
    diag = [jaccard(siren_neurons[l], zhao_neurons[l]) for l in common]
    chance = [chance_jaccard(len(siren_neurons[l]), len(zhao_neurons[l]))
              for l in common]
    ax = axes[1]
    ax.plot(common, diag, marker="o", markersize=4, color="tab:purple",
            label="observed")
    ax.plot(common, chance, color="gray", linestyle=":", label="chance level")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Same-layer overlap: SIREN (mlpneuron) vs zhao/svea (ffn)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ffn_space_jaccard_siren_vs_zhao.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
    return common, diag, chance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siren_probes", required=True,
                        help="probes pkl from the mlpneuron_mean run "
                             "(contains best_probes with layer{N}_mlpneuron_mean keys)")
    parser.add_argument("--svea", required=True,
                        help="results/probes-svea.pkl (Zhao reproduction DataFrame)")
    parser.add_argument("--pooling_type", default="mlpneuron_mean")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    siren_neurons, _ = load_siren(args.siren_probes, args.pooling_type,
                                  args.threshold)
    if not siren_neurons:
        raise SystemExit(
            f"No 'layer{{N}}_{args.pooling_type}' probes found in "
            f"{args.siren_probes} — was this pkl produced by a "
            f"{args.pooling_type} training run?")
    zhao_neurons = load_zhao_ffn(args.svea)

    plot_counts(siren_neurons, zhao_neurons, args.output_dir)
    common, diag, chance = plot_jaccard(siren_neurons, zhao_neurons,
                                        args.threshold, args.output_dir)

    # Global overlap over (layer, neuron) pairs — the single-number summary.
    siren_pairs = {(l, i) for l, s in siren_neurons.items() for i in s}
    zhao_pairs = {(l, i) for l, s in zhao_neurons.items() for i in s}
    print(f"\nSIREN selected {len(siren_pairs)} (layer, neuron) pairs, "
          f"zhao/svea ffn selected {len(zhao_pairs)}")
    print(f"Global (layer, neuron) Jaccard: {jaccard(siren_pairs, zhao_pairs):.4f}")
    if common:
        print(f"Same-layer Jaccard: mean={np.mean(diag):.4f} "
              f"max={np.max(diag):.4f} (chance mean={np.mean(chance):.4f})")


if __name__ == "__main__":
    main()

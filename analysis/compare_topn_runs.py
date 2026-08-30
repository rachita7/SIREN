"""
Compare the exact top-N neuron selections of TWO training runs of the same
model/pooling (e.g. the clean standard run trained with the mlpneuron C grid
vs. one trained with the original residual C grid), to check how sensitive
the selection is to hyperparameters.

For each budget N the script reports:
  - Jaccard overlap of the two runs' top-N (layer, neuron) pair sets
    (with the chance level for reference: two random size-N subsets of the
    32x14336 space overlap almost not at all)
  - Pearson correlation of the two runs' per-layer count profiles
    (do the selections at least concentrate in the same layers?)

plus a per-layer counts plot (run A solid, run B dashed, one color per N)
and CSVs of the summary and per-layer counts.

Usage:

    python analysis/compare_topn_runs.py --model llama3-8b-instruct \
        --pooling_type mlpneuron_mean \
        --suffix_a=-std-mlpneuron_mean-clean \
        --suffix_b=-std-mlpneuron_mean-clean-origC \
        --targets 459 2294 4588 9175
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_siren_neurons import load_probes
from export_topn_neurons import entry_thresholds


def ranked_pairs(probes, layers):
    """All (layer, neuron) pairs in global selection order (see
    export_topn_neurons: sorted by cumulative-importance entry threshold)."""
    pairs = []
    for layer_idx in layers:
        order, prefix = entry_thresholds(probes[layer_idx])
        pairs.extend((float(prefix[r]), r, layer_idx, int(order[r]))
                     for r in range(len(order)))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return [(layer, idx) for _, _, layer, idx in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="mlpneuron_mean")
    parser.add_argument("--suffix_a", type=str, required=True)
    parser.add_argument("--suffix_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default=None,
                        help="Legend label for run A (default: its suffix).")
    parser.add_argument("--label_b", type=str, default=None)
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[459, 2294, 4588, 9175])
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "train", "probes"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    label_a = args.label_a or args.suffix_a.lstrip("-")
    label_b = args.label_b or args.suffix_b.lstrip("-")

    layers_a, probes_a = load_probes(args.model, args.probes_dir,
                                     args.pooling_type, args.suffix_a)
    layers_b, probes_b = load_probes(args.model, args.probes_dir,
                                     args.pooling_type, args.suffix_b)
    layers = sorted(set(layers_a) & set(layers_b))
    order_a = ranked_pairs(probes_a, layers)
    order_b = ranked_pairs(probes_b, layers)
    total = len(order_a)

    tag = f"{args.model}_{args.pooling_type}_compare_{label_a}_vs_{label_b}"
    budgets = sorted(args.targets)
    summary = []
    counts = {n: {} for n in budgets}  # {n: {run_label: {layer: count}}}

    print(f"Comparing top-N selections over {len(layers)} layers "
          f"({total} pairs total):\n")
    for n in budgets:
        set_a, set_b = set(order_a[:n]), set(order_b[:n])
        inter = len(set_a & set_b)
        jac = inter / len(set_a | set_b)
        # expected Jaccard of two independent random size-n subsets
        exp_inter = n * n / total
        chance = exp_inter / (2 * n - exp_inter)

        cnt_a = {l: 0 for l in layers}
        cnt_b = {l: 0 for l in layers}
        for l, _ in set_a:
            cnt_a[l] += 1
        for l, _ in set_b:
            cnt_b[l] += 1
        counts[n] = {label_a: cnt_a, label_b: cnt_b}
        prof_a = np.array([cnt_a[l] for l in layers], dtype=float)
        prof_b = np.array([cnt_b[l] for l in layers], dtype=float)
        layer_corr = float(np.corrcoef(prof_a, prof_b)[0, 1])

        print(f"top{n:>6}: Jaccard={jac:.3f} (chance {chance:.4f}), "
              f"|A∩B|={inter}, layer-profile corr={layer_corr:.3f}")
        summary.append([n, inter, f"{jac:.4f}", f"{chance:.6f}",
                        f"{layer_corr:.4f}"])

    csv_path = os.path.join(args.output_dir, f"{tag}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_n", "intersection", "jaccard",
                         "chance_jaccard", "layer_profile_corr"])
        writer.writerows(summary)
    print(f"\nSaved {csv_path}")

    counts_csv = os.path.join(args.output_dir, f"{tag}_layer_counts.csv")
    with open(counts_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"]
                        + [f"top{n}_{lab}" for n in budgets
                           for lab in (label_a, label_b)])
        for l in layers:
            writer.writerow([l] + [counts[n][lab][l] for n in budgets
                                   for lab in (label_a, label_b)])
    print(f"Saved {counts_csv}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(budgets)))
    for color, n in zip(colors, budgets):
        ax.plot(layers, [counts[n][label_a][l] for l in layers], color=color,
                marker="o", markersize=3, label=f"top {n} ({label_a})")
        ax.plot(layers, [counts[n][label_b][l] for l in layers], color=color,
                marker="s", markersize=3, linestyle="--",
                label=f"top {n} ({label_b})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("# selected safety neurons")
    ax.set_title(f"Top-N selections per layer: {label_a} (solid) vs "
                 f"{label_b} (dashed)\n{args.model} ({args.pooling_type})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{tag}_layer_counts.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()

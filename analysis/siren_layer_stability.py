"""
Layer-level stability of SIREN selections across three independent trainings
(train/run_stability_siren.sh -> probes/..._general_probes-stability-
{pooling}-{trainsplit|split}{1,2,3}.pkl).

Rationale: SIREN's selection is created by probe training, so stability is
tested by re-training on disjoint data thirds. Because SIREN's atomic units
are residual-stream dimensions (not physical neurons), the comparison is
kept at the LAYER level -- the method's interpretable output is its layer
profile: where in the network's depth the safety signal concentrates.

For each run and layer this script computes:
  - the probe's validation / test macro-F1 (is the layer informative?)
  - the number of features selected at a cumulative-importance threshold
    (how concentrated is the layer's signal?), normalized to a per-run
    density so the three profiles are directly overlayable -- and
    comparable with the Zhao-side layer histograms.

Stability statistics across the three runs:
  - Spearman correlation between each pair of runs' layer profiles
    (for both the F1 profile and the selection-share profile)
  - Jaccard overlap of the top-k layers per run (default k=8)

Usage (after the three trainings):

    python analysis/siren_layer_stability.py --model llama3-8b-instruct \
        --pooling_type residual_mean --threshold 0.9
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import argparse
import csv
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from plot_all_methods_overlap import CPUUnpickler, select_cumulative


def load_run(path, pooling_type, threshold):
    """{layer: (val_f1, test_f1, num_selected)} for one training run."""
    with open(path, "rb") as f:
        data = CPUUnpickler(f).load()
    out = {}
    suffix = f"_{pooling_type}"
    for key, entry in data["best_probes"].items():
        if not key.endswith(suffix):
            continue
        layer = int(key[len("layer"):-len(suffix)])
        w = entry["probe"].get_feature_importance()
        out[layer] = (float(entry["val_f1"]), float(entry["test_f1"]),
                      len(select_cumulative(w, threshold)))
    if not out:
        raise ValueError(f"No 'layer{{N}}{suffix}' probes in {path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Cumulative-importance threshold for per-layer "
                             "selection counts.")
    parser.add_argument("--top_k_layers", type=int, default=8)
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "train", "probes"))
    parser.add_argument("--suffix_template", type=str,
                        default="-stability-{pooling}-trainsplit{i}",
                        help="Suffix of each run's probes pkl; {pooling} and "
                             "{i} are substituted. Use ...-split{i} for the "
                             "test-split variant.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for i in (1, 2, 3):
        suffix = args.suffix_template.format(pooling=args.pooling_type, i=i)
        path = os.path.join(args.probes_dir,
                            f"{args.model}_general_probes{suffix}.pkl")
        run = load_run(path, args.pooling_type, args.threshold)
        runs.append(run)
        vals = [v[0] for v in run.values()]
        print(f"split {i}: {len(run)} layers, mean val F1 = {np.mean(vals):.3f} "
              f"({os.path.basename(path)})")

    layers = sorted(set(runs[0]) & set(runs[1]) & set(runs[2]))
    val_f1 = np.array([[r[l][0] for l in layers] for r in runs])
    test_f1 = np.array([[r[l][1] for l in layers] for r in runs])
    counts = np.array([[r[l][2] for l in layers] for r in runs], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------- stats
    print(f"\nLayer-profile stability across the 3 runs "
          f"(threshold={args.threshold}):")
    for name, profile in (("val F1", val_f1), ("selection share", shares)):
        rhos = [spearmanr(profile[a], profile[b]).statistic
                for a, b in combinations(range(3), 2)]
        print(f"  Spearman({name} profiles): "
              + " / ".join(f"{r:.3f}" for r in rhos)
              + f"  (mean {np.mean(rhos):.3f})")

    k = args.top_k_layers
    for name, profile in (("val F1", val_f1), ("selection share", shares)):
        tops = [set(np.argsort(profile[r])[::-1][:k]) for r in range(3)]
        jacs = [len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                for a, b in combinations(range(3), 2)]
        stable = set.intersection(*tops)
        print(f"  top-{k} layers by {name}: pairwise Jaccard "
              + " / ".join(f"{j:.3f}" for j in jacs)
              + f"; stable in all 3 runs: {sorted(layers[i] for i in stable)}")

    # ---------------------------------------------------------------- save
    tag = f"{args.model}_{args.pooling_type}_layer_stability"
    csv_path = os.path.join(args.output_dir, f"{tag}_t{args.threshold}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"]
                        + [f"val_f1_run{r+1}" for r in range(3)]
                        + [f"test_f1_run{r+1}" for r in range(3)]
                        + [f"num_selected_run{r+1}" for r in range(3)])
        for j, l in enumerate(layers):
            writer.writerow([l] + [f"{val_f1[r][j]:.4f}" for r in range(3)]
                            + [f"{test_f1[r][j]:.4f}" for r in range(3)]
                            + [int(counts[r][j]) for r in range(3)])
    print(f"\nSaved {csv_path}")

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in range(3):
        axes[0].plot(layers, val_f1[r], marker="o", markersize=3, alpha=0.7,
                     label=f"run {r+1}")
        axes[1].plot(layers, shares[r], marker="o", markersize=3, alpha=0.7,
                     label=f"run {r+1}")
    axes[0].plot(layers, val_f1.mean(axis=0), color="black", linewidth=2,
                 label="mean")
    axes[1].plot(layers, shares.mean(axis=0), color="black", linewidth=2,
                 label="mean")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probe val macro-F1")
    axes[0].set_title(f"Layer-wise probe performance across 3 trainings\n"
                      f"{args.model} ({args.pooling_type})")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Share of selected features in layer")
    axes[1].set_title(f"Layer distribution of selection "
                      f"(threshold={args.threshold})")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{tag}_t{args.threshold}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()

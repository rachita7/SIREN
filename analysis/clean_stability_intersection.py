"""
Clean-stability intersection analysis for SIREN.

Input: three probe trainings on the pre-made thirds of the cleaned train set
(train/run_clean_stability_siren.sh -> probes/{model}_general_probes-
clean_stability-{pooling}-split{1,2,3}.pkl; val/test kept their normal roles).

This script:
  1. reports each run's per-layer validation/test macro-F1 (test = the shared
     standard test files, so the three models are directly comparable), and
  2. intersects the three runs' neuron selections and exports the top
     N in {459, 2294, 4588, 9175} (0.1/0.5/1/2% of the 32x14336 FFN space).

Intersection-tuning without search: within one run, every (layer, neuron)
pair has a global rank -- pairs sorted by the cumulative-importance "entry
threshold" of their layer probe (see export_topn_neurons.py). A pair is in
the intersection of the three runs' top-M selections iff its WORST rank
across runs is < M. Sorting pairs by worst rank therefore yields, for any N,
exactly the N pairs most robustly selected by all three runs, plus the
per-run budget M this corresponds to.

Usage (after the three trainings):

    python analysis/clean_stability_intersection.py --model llama3-8b-instruct \
        --pooling_type mlpneuron_mean --targets 459 2294 4588 9175
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_siren_neurons import CPUUnpickler
from export_topn_neurons import entry_thresholds


def load_run(path, pooling_type):
    """{layer: {"probe": ..., "val_f1": ..., "test_f1": ...}} for one run."""
    with open(path, "rb") as f:
        data = CPUUnpickler(f).load()
    out = {}
    key_suffix = f"_{pooling_type}"
    for key, entry in data["best_probes"].items():
        if not key.endswith(key_suffix):
            continue
        layer = int(key[len("layer"):-len(key_suffix)])
        out[layer] = {"probe": entry["probe"],
                      "val_f1": float(entry["val_f1"]),
                      "test_f1": float(entry["test_f1"])}
    if not out:
        raise ValueError(f"No 'layer{{N}}{key_suffix}' probes in {path}")
    return out


def global_ranks(run):
    """{(layer, neuron): rank} -- pairs ordered by entry threshold (see
    export_topn_neurons), i.e. the order a per-layer threshold sweep selects."""
    pairs = []
    for layer, entry in run.items():
        order, prefix = entry_thresholds(entry["probe"])
        pairs.extend((float(prefix[r]), r, layer, int(order[r]))
                     for r in range(len(order)))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return {(layer, idx): rank for rank, (_, _, layer, idx) in enumerate(pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="mlpneuron_mean")
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[459, 2294, 4588, 9175])
    parser.add_argument("--suffix_template", type=str,
                        default="-clean_stability-{pooling}-split{i}")
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "train", "probes"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.model}_{args.pooling_type}_clean_stability"

    # ------------------------------------------------ load runs + test results
    runs = []
    for i in (1, 2, 3):
        suffix = args.suffix_template.format(pooling=args.pooling_type, i=i)
        path = os.path.join(args.probes_dir,
                            f"{args.model}_general_probes{suffix}.pkl")
        run = load_run(path, args.pooling_type)
        runs.append(run)
        val = np.mean([e["val_f1"] for e in run.values()])
        test = np.mean([e["test_f1"] for e in run.values()])
        print(f"split {i}: {len(run)} layers, mean val F1 = {val:.3f}, "
              f"mean test F1 = {test:.3f}  ({os.path.basename(path)})")

    layers = sorted(set(runs[0]) & set(runs[1]) & set(runs[2]))
    f1_csv = os.path.join(args.output_dir, f"{tag}_probe_f1.csv")
    with open(f1_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"]
                        + [f"val_f1_split{r+1}" for r in range(3)]
                        + [f"test_f1_split{r+1}" for r in range(3)])
        for l in layers:
            writer.writerow([l]
                            + [f"{runs[r][l]['val_f1']:.4f}" for r in range(3)]
                            + [f"{runs[r][l]['test_f1']:.4f}" for r in range(3)])
    print(f"Saved {f1_csv}")

    # ------------------------------------------- worst-rank tuned intersection
    ranks = [global_ranks(run) for run in runs]
    all_pairs = set(ranks[0])
    scored = sorted(
        ((max(rk[p] for rk in ranks),          # worst rank -> intersection budget
          float(np.mean([rk[p] for rk in ranks])),  # tie-break: mean rank
          p)
         for p in all_pairs),
        key=lambda x: (x[0], x[1], x[2]))

    summary_rows = []
    counts_by_n = {}  # {target_n: {layer: count}} for the plots
    for n in sorted(args.targets):
        if n > len(scored):
            print(f"top{n}: only {len(scored)} pairs exist, skipping")
            continue
        chosen = scored[:n]
        budget_m = chosen[-1][0] + 1  # per-run top-M whose intersection covers these N
        raw_inter = sum(1 for w, _, _ in scored if w < budget_m)

        per_layer = {}
        for _, _, (layer, idx) in chosen:
            per_layer.setdefault(layer, []).append(idx)
        counts_by_n[n] = {l: len(per_layer.get(l, [])) for l in layers}
        out = {f"layer{l}": per_layer[l] for l in sorted(per_layer)}
        path = os.path.join(args.output_dir,
                            f"{tag}_intersection_top{n}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        counts = sorted(((l, len(v)) for l, v in per_layer.items()),
                        key=lambda kv: -kv[1])
        print(f"top{n}: per-run budget M={budget_m} "
              f"(intersection size at M: {raw_inter}), "
              f"{len(per_layer)}/{len(layers)} layers, "
              f"biggest layers: {counts[:5]}")
        print(f"  saved {path}")
        summary_rows.append([n, budget_m, raw_inter, len(per_layer)])

    summary_csv = os.path.join(args.output_dir, f"{tag}_intersection_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_n", "per_run_budget_m",
                         "intersection_size_at_m", "layers_used"])
        writer.writerows(summary_rows)
    print(f"Saved {summary_csv}")

    # ------------------------------------------------------------------ plots
    # (1) probe F1 of the 3 split models (left) + per-layer intersection neuron
    #     counts, one curve per budget (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in range(3):
        axes[0].plot(layers, [runs[r][l]["val_f1"] for l in layers], marker="o",
                     markersize=3, label=f"split {r+1} (val)")
        axes[0].plot(layers, [runs[r][l]["test_f1"] for l in layers], marker="s",
                     markersize=3, linestyle="--", alpha=0.6,
                     label=f"split {r+1} (test)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probe macro-F1")
    axes[0].set_title(f"Layer-wise probe performance, 3 clean train splits\n"
                      f"{args.model} ({args.pooling_type})")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for n, cnts in counts_by_n.items():
        axes[1].plot(layers, [cnts[l] for l in layers], marker="o",
                     markersize=3, label=f"top {n}")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("# intersection safety neurons")
    axes[1].set_title("Intersection of the 3 runs' selections, per layer")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    probe_plot = os.path.join(args.output_dir, f"{tag}_layer_probes.png")
    plt.savefig(probe_plot, dpi=200)
    plt.close(fig)
    print(f"Saved {probe_plot}")

    # (2) standalone version of the intersection-counts panel
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, cnts in counts_by_n.items():
        ax.plot(layers, [cnts[l] for l in layers], marker="o", markersize=3,
                label=f"top {n}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("# intersection safety neurons")
    ax.set_title(f"SIREN clean-stability intersection neurons per layer\n"
                 f"{args.model} ({args.pooling_type})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    counts_plot = os.path.join(args.output_dir, f"{tag}_intersection_counts.png")
    plt.savefig(counts_plot, dpi=200)
    plt.close(fig)
    print(f"Saved {counts_plot}")


if __name__ == "__main__":
    main()

"""
SIREN stability selection, mirroring the Wang/Zhao reproduction's protocol:

  1. Three independent probe trainings, one per standardized test-set third
     (train/run_stability_siren.sh -> probes/..._general_probes-stability-
     {pooling}-split{1,2,3}.pkl).
  2. Per run, rank all (layer, dim) features globally by per-layer-normalized
     |probe weight| (each layer's weights are divided by their sum, so a
     feature's score is "share of its layer's importance mass").
  3. For each target N (2500 / 5000 / 10000): find the smallest per-run
     top-K such that the intersection of the three runs' top-K sets reaches
     N ("reverse engineering" per the protocol), then trim the intersection
     to exactly N by best (worst-case) rank.

Outputs, per pooling type:
  - JSON with the stable feature sets per target N (per-layer index lists)
  - CSV with per-layer stable-feature counts per target N
  - a per-layer density histogram (share of stable features per layer),
    directly comparable to the Zhao-side histogram plots
  - printed stats: required per-run K, stability rate N/K, pairwise Jaccard
    between the runs' top-K selections

NOTE on terminology: for residual_mean these are residual-stream DIMENSIONS
(shared representation coordinates), not physical neurons; for
mlpneuron_mean they are actual FFN neurons in the same space as Zhao's `ffn`
component. The protocol and stats are identical either way.

Usage (after the three trainings):

    python analysis/siren_stability_neurons.py --model llama3-8b-instruct \
        --pooling_type residual_mean --targets 2500 5000 10000
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

from plot_all_methods_overlap import CPUUnpickler


def load_run(path, pooling_type):
    """{layer: normalized |weight| vector} + {layer: val_f1} for one run."""
    with open(path, "rb") as f:
        data = CPUUnpickler(f).load()
    weights, val_f1 = {}, {}
    suffix = f"_{pooling_type}"
    for key, entry in data["best_probes"].items():
        if not key.endswith(suffix):
            continue
        layer = int(key[len("layer"):-len(suffix)])
        w = entry["probe"].get_feature_importance().astype(np.float64)
        total = w.sum()
        weights[layer] = w / total if total > 0 else w
        val_f1[layer] = float(entry["val_f1"])
    if not weights:
        raise ValueError(f"No 'layer{{N}}{suffix}' probes in {path}")
    return weights, val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[2500, 5000, 10000])
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "train", "probes"))
    parser.add_argument("--suffix_template", type=str,
                        default="-stability-{pooling}-split{i}",
                        help="Suffix of each run's probes pkl; {pooling} and "
                             "{i} are substituted.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------- load
    runs = []
    for i in (1, 2, 3):
        suffix = args.suffix_template.format(pooling=args.pooling_type, i=i)
        path = os.path.join(args.probes_dir,
                            f"{args.model}_general_probes{suffix}.pkl")
        weights, val_f1 = load_run(path, args.pooling_type)
        runs.append(weights)
        print(f"split {i}: {len(weights)} layers, "
              f"mean val F1 = {np.mean(list(val_f1.values())):.3f} "
              f"({os.path.basename(path)})")

    layers = sorted(set(runs[0]) & set(runs[1]) & set(runs[2]))
    dim = len(runs[0][layers[0]])
    total_features = len(layers) * dim
    print(f"\nFeature space: {len(layers)} layers x {dim} dims "
          f"= {total_features} features")

    # Flat score matrix (num_runs, total_features); flat index -> (layer, dim)
    flat = np.stack([
        np.concatenate([run[l] for l in layers]) for run in runs
    ])

    # rank[r, j] = position of feature j in run r's descending-score order
    ranks = np.empty_like(flat, dtype=np.int64)
    for r in range(flat.shape[0]):
        order = np.argsort(flat[r])[::-1]
        ranks[r, order] = np.arange(total_features)

    # Feature j is in ALL runs' top-K  <=>  max-over-runs rank < K.
    max_rank = ranks.max(axis=0)
    mean_rank = ranks.mean(axis=0)

    results = {"model": args.model, "pooling_type": args.pooling_type,
               "num_layers": len(layers), "dim_per_layer": dim,
               "note": ("indices are residual-stream dimensions for "
                        "residual_mean, FFN neurons for mlpneuron_mean"),
               "targets": {}}
    per_layer_counts = {}

    for target in args.targets:
        if target > total_features:
            print(f"target {target} exceeds feature space; skipping")
            continue
        # Smallest per-run K whose triple intersection reaches the target:
        # the target-th smallest max_rank (+1 to convert rank to set size).
        sorted_max = np.sort(max_rank)
        K = int(sorted_max[target - 1]) + 1
        inter_size = int(np.sum(max_rank < K))

        # Exact-N stable set: best worst-case rank, ties by mean rank.
        order = np.lexsort((mean_rank, max_rank))
        stable = order[:target]

        stable_per_layer = {l: [] for l in layers}
        for j in stable:
            stable_per_layer[layers[j // dim]].append(int(j % dim))
        counts = {l: len(v) for l, v in stable_per_layer.items()}
        per_layer_counts[target] = counts

        # Pairwise Jaccard between the runs' top-K selections (stability).
        tops = [set(np.flatnonzero(ranks[r] < K)) for r in range(3)]
        pair_j = [len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                  for a, b in ((0, 1), (0, 2), (1, 2))]

        results["targets"][str(target)] = {
            "per_run_top_k": K,
            "intersection_size_at_k": inter_size,
            "stability_rate": target / K,
            "pairwise_jaccard_at_k": [round(j, 4) for j in pair_j],
            "stable_neurons": {f"layer{l}": sorted(v)
                               for l, v in stable_per_layer.items() if v},
        }
        print(f"\ntarget N={target}: per-run top-K={K} "
              f"(|intersection|={inter_size}, stability rate N/K={target/K:.3f})")
        print(f"  pairwise Jaccard of the three top-K sets: "
              f"{pair_j[0]:.3f} / {pair_j[1]:.3f} / {pair_j[2]:.3f}")

    # ---------------------------------------------------------------- save
    tag = f"{args.model}_{args.pooling_type}_stability"
    json_path = os.path.join(args.output_dir, f"{tag}_neurons.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")

    csv_path = os.path.join(args.output_dir, f"{tag}_layer_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        targets = sorted(per_layer_counts)
        writer.writerow(["layer"] + [f"stable_n{t}" for t in targets])
        for l in layers:
            writer.writerow([l] + [per_layer_counts[t].get(l, 0) for t in targets])
    print(f"Saved {csv_path}")

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in sorted(per_layer_counts):
        counts = per_layer_counts[target]
        vals = np.array([counts.get(l, 0) for l in layers], dtype=float)
        ax.plot(layers, vals / vals.sum(), marker="o", markersize=3.5,
                label=f"stable N={target}")
    ax.axhline(1.0 / len(layers), color="gray", linestyle=":",
               label="uniform across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of stable features in layer")
    ax.set_title(f"Layer distribution of stability-selected features\n"
                 f"{args.model} ({args.pooling_type}, intersection of 3 test splits)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{tag}_layer_hist.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()

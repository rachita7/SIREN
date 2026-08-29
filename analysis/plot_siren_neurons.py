"""
Per-layer SIREN safety-neuron plots for a single backbone.

Mirrors the "detected neurons per layer, by method" figures, but for SIREN:
SIREN fits one L1 linear probe per layer, ranks that layer's features
("neurons") by |weight|, and keeps the top ones until their cumulative
importance reaches a threshold. This script visualizes, for one model:

  (a) how many safety neurons SIREN selects at each layer (one curve per
      cumulative-importance threshold), and
  (b) how distinct those per-layer neuron sets are, as a layer x layer Jaccard
      overlap heatmap (residual/mlp features share one coordinate space across
      layers, so the same index means the same neuron in every layer).

Reads train/probes/<model>_general_probes.pkl (written by training). CUDA
tensors are mapped to CPU, so no GPU is needed:

    python analysis/plot_siren_neurons.py --model llama3-8b-instruct \
        --thresholds 0.6 0.8 0.9 --heatmap_threshold 0.9
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import io
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import MODEL_CONFIGS


class CPUUnpickler(pickle.Unpickler):
    """Map CUDA tensors to CPU so pickled probes load on a GPU-less machine."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


def select_salient_neurons(probe, threshold):
    """Top neurons by |weight| until cumulative importance >= threshold*total.
    Identical to the selection used in training and plot_layer_probes.py."""
    weights = probe.get_feature_importance()
    total = np.sum(weights)
    order = np.argsort(weights)[::-1]
    selected = []
    cumulative = 0.0
    for idx in order:
        selected.append(int(idx))
        cumulative += weights[idx]
        if total > 0 and cumulative >= threshold * total:
            break
    return selected


def load_probes(model_name, probes_dir, pooling_type, suffix=""):
    probe_path = os.path.join(probes_dir, f"{model_name}_general_probes{suffix}.pkl")
    if not os.path.exists(probe_path):
        raise FileNotFoundError(
            f"{probe_path} not found. Run training first "
            f"(train/run_standard_siren.sh {model_name})."
        )
    with open(probe_path, "rb") as f:
        data = CPUUnpickler(f).load()
    best_probes = data["best_probes"]
    num_layers = MODEL_CONFIGS[model_name]["num_layers"]

    layers, probes = [], {}
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            layers.append(layer_idx)
            probes[layer_idx] = best_probes[key]["probe"]
    if not layers:
        raise ValueError(
            f"No probes for pooling_type='{pooling_type}' in {probe_path}. "
            f"Available keys look like: {list(best_probes)[:3]}"
        )
    return layers, probes


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.8, 0.9])
    parser.add_argument("--top_n", type=int, nargs="+", default=None,
                        help="Instead of cumulative-importance thresholds, plot "
                             "per-layer counts under EXACT global neuron budgets "
                             "(e.g. --top_n 459 2294 4588 9175): the N globally "
                             "top-ranked (layer, neuron) pairs, one curve per N.")
    parser.add_argument("--heatmap_threshold", type=float, default=0.9,
                        help="Threshold used for the layer x layer Jaccard heatmap.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Training-run suffix in the pkl filename, e.g. "
                             "'-mlpneuron_mean' (see run_hh_siren.sh OUTPUT_SUFFIX).")
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "train", "probes"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--log_y", action="store_true",
                        help="Log-scale the per-layer neuron-count axis.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    layers, probes = load_probes(args.model, args.probes_dir, args.pooling_type, args.suffix)

    if args.top_n:
        # Exact-budget mode: rank ALL (layer, neuron) pairs globally by their
        # cumulative-importance entry threshold and, for each budget N, count
        # how many of the first N pairs fall in each layer.
        from export_topn_neurons import entry_thresholds  # lazy: avoids import cycle
        pairs = []
        for layer_idx in layers:
            order, prefix = entry_thresholds(probes[layer_idx])
            pairs.extend((float(prefix[r]), r, layer_idx) for r in range(len(order)))
        pairs.sort(key=lambda p: (p[0], p[1]))

        budgets = sorted(args.top_n)
        counts = {n: {l: 0 for l in layers} for n in budgets}
        for rank, (_, _, layer_idx) in enumerate(pairs[:max(budgets)]):
            for n in budgets:
                if rank < n:
                    counts[n][layer_idx] += 1

        fig, ax = plt.subplots(figsize=(9, 5))
        for n in budgets:
            ax.plot(layers, [counts[n][l] for l in layers], marker="o",
                    markersize=3, label=f"top {n}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("# selected safety neurons")
        ax.set_title(f"SIREN safety neurons per layer at fixed global budgets\n"
                     f"{args.model} ({args.pooling_type})")
        if args.log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        n_tag = "-".join(str(n) for n in budgets)
        plot_path = os.path.join(
            args.output_dir,
            f"{args.model}_siren_neurons_{args.pooling_type}{args.suffix}_top{n_tag}.png")
        plt.savefig(plot_path, dpi=200)
        print(f"Saved {plot_path}")

        csv_path = os.path.join(
            args.output_dir,
            f"{args.model}_{args.pooling_type}{args.suffix}_neuron_counts_top{n_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer"] + [f"num_selected_top{n}" for n in budgets])
            for l in layers:
                writer.writerow([l] + [counts[n][l] for n in budgets])
        print(f"Saved {csv_path}")
        return

    # counts[threshold] = list of #selected per layer; selected_at_heatmap keeps
    # the actual index sets for the overlap heatmap.
    counts = {t: [] for t in args.thresholds}
    selected_at_heatmap = {}
    for layer_idx in layers:
        probe = probes[layer_idx]
        for t in args.thresholds:
            counts[t].append(len(select_salient_neurons(probe, t)))
        selected_at_heatmap[layer_idx] = select_salient_neurons(probe, args.heatmap_threshold)

    # --- Save the actual selected neuron indices per layer (heatmap threshold) ---
    idx_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.pooling_type}{args.suffix}_selected_neurons_t{args.heatmap_threshold}.json")
    import json
    with open(idx_path, "w") as f:
        json.dump({f"layer{k}": v for k, v in selected_at_heatmap.items()}, f, indent=2)
    print(f"Saved {idx_path}")

    # --- Save per-layer counts CSV ---
    csv_path = os.path.join(args.output_dir, f"{args.model}_{args.pooling_type}{args.suffix}_neuron_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"num_selected_t{t}" for t in args.thresholds])
        for i, layer_idx in enumerate(layers):
            writer.writerow([layer_idx] + [counts[t][i] for t in args.thresholds])
    print(f"Saved {csv_path}")

    # --- Figure: (a) counts per layer, (b) cross-layer Jaccard heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for t in args.thresholds:
        axes[0].plot(layers, counts[t], marker="o", markersize=3, label=f"threshold={t}")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("# selected safety neurons")
    axes[0].set_title(f"SIREN safety neurons per layer\n{args.model} ({args.pooling_type})")
    if args.log_y:
        axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    n = len(layers)
    jac = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = jaccard(selected_at_heatmap[layers[i]], selected_at_heatmap[layers[j]])
            jac[i, j] = jac[j, i] = v
    im = axes[1].imshow(jac, origin="lower", aspect="auto", vmin=0, vmax=1,
                        extent=[layers[0], layers[-1], layers[0], layers[-1]])
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Layer")
    axes[1].set_title(f"Cross-layer neuron overlap (Jaccard, t={args.heatmap_threshold})")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Jaccard")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{args.model}_siren_neurons_{args.pooling_type}{args.suffix}.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved {plot_path}")

    off_diag = jac[~np.eye(n, dtype=bool)]
    print(f"Mean off-diagonal Jaccard (how much layers share neurons): {off_diag.mean():.3f}")
    print(f"Total distinct neurons selected across all layers (t={args.heatmap_threshold}): "
          f"{len(set().union(*selected_at_heatmap.values()))}")


if __name__ == "__main__":
    main()

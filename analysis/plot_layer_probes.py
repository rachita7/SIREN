"""
Layer-wise analysis of SIREN safety-neuron probes.

For each trained backbone this script plots, per layer:
  1. the L1 linear probe's validation/test macro-F1 (where harmfulness is
     linearly decodable), and
  2. the number and share of safety neurons selected at a cumulative-importance
     threshold (where the safety signal is concentrated).

Overlaying llama3-8b-sft and llama3-8b-instruct shows how alignment shifts the
layer distribution of safety-relevant neurons (mid vs. late layers).

Run on a machine with a GPU (the pickled probes contain CUDA tensors), after
train/run_hh_siren.sh has finished:

    python analysis/plot_layer_probes.py --models llama3-8b-sft llama3-8b-instruct --threshold 0.9
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

from utils.config import MODEL_CONFIGS


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


def load_layer_stats(model_name, probes_dir, pooling_type, threshold):
    probe_path = os.path.join(probes_dir, f"{model_name}_general_probes.pkl")
    with open(probe_path, "rb") as f:
        data = pickle.load(f)
    best_probes = data["best_probes"]

    num_layers = MODEL_CONFIGS[model_name]["num_layers"]
    rows = []
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key not in best_probes:
            continue
        entry = best_probes[key]
        selected = select_salient_neurons(entry["probe"], threshold)
        rows.append({
            "model": model_name,
            "layer": layer_idx,
            "val_f1": entry["val_f1"],
            "test_f1": entry["test_f1"],
            "num_selected": len(selected),
            "selected_neurons": selected,
        })
    df = pd.DataFrame(rows)
    df["selected_share"] = df["num_selected"] / df["num_selected"].sum()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "train", "probes"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for model_name in args.models:
        df = load_layer_stats(model_name, args.probes_dir, args.pooling_type, args.threshold)
        all_stats.append(df)
        csv_path = os.path.join(args.output_dir, f"{model_name}_{args.pooling_type}_layer_stats.csv")
        df.drop(columns=["selected_neurons"]).to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for df in all_stats:
        model_name = df["model"].iloc[0]
        axes[0].plot(df["layer"], df["val_f1"], marker="o", markersize=3, label=f"{model_name} (val)")
        axes[0].plot(df["layer"], df["test_f1"], marker="s", markersize=3, linestyle="--", alpha=0.6,
                     label=f"{model_name} (test)")
        axes[1].plot(df["layer"], df["selected_share"], marker="o", markersize=3, label=model_name)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probe macro-F1")
    axes[0].set_title(f"Layer-wise probe performance ({args.pooling_type})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Share of selected safety neurons")
    axes[1].set_title(f"Safety-neuron distribution (threshold={args.threshold})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"layer_probes_{args.pooling_type}_t{args.threshold}.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved {plot_path}")

    if len(all_stats) == 2:
        a, b = all_stats
        merged = a.merge(b, on="layer", suffixes=("_a", "_b"))
        jaccards = []
        for _, row in merged.iterrows():
            sa, sb = set(row["selected_neurons_a"]), set(row["selected_neurons_b"])
            union = sa | sb
            jaccards.append(len(sa & sb) / len(union) if union else 0.0)
        merged["jaccard"] = jaccards
        jac_path = os.path.join(args.output_dir, f"neuron_overlap_{args.pooling_type}_t{args.threshold}.csv")
        merged[["layer", "num_selected_a", "num_selected_b", "jaccard"]].to_csv(jac_path, index=False)
        print(f"Saved per-layer neuron overlap (Jaccard) between "
              f"{a['model'].iloc[0]} and {b['model'].iloc[0]}: {jac_path}")


if __name__ == "__main__":
    main()

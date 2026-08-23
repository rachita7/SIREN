"""
Export SIREN selected-neuron JSONs with an EXACT total number of
(layer, neuron) pairs (e.g. 2500 / 5000 / 10000), for comparison with methods
that report fixed neuron budgets.

How it stays faithful to SIREN's selection rule: per layer, SIREN keeps
neurons in descending |weight| order until their cumulative importance reaches
a threshold t (same t for every layer). Each neuron therefore has an "entry
threshold": the normalized importance mass that precedes it in its own layer's
ranking -- it is selected iff its entry threshold < t. Sorting ALL
(layer, neuron) pairs by entry threshold and keeping the first N reproduces
exactly the set a per-layer threshold sweep would give, while hitting N
precisely (a raw threshold can only jump between totals). The effective
threshold that N corresponds to is printed and stored in the JSON filename's
sidecar summary.

Output format matches plot_siren_neurons.py's selected_neurons JSON:
{"layer0": [idx, ...], ...} with indices in descending per-layer importance.

Usage (on the machine that has the probes pkl):

    python analysis/export_topn_neurons.py --model llama3-8b-instruct \
        --pooling_type mlpneuron_mean --suffix=-std-mlpneuron_mean \
        --targets 2500 5000 10000
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import argparse
import json

import numpy as np

from plot_siren_neurons import load_probes


def entry_thresholds(probe):
    """Per neuron: (entry threshold, importance rank) in its layer.

    entry[i] = normalized importance mass strictly before neuron i in the
    layer's descending-|weight| ranking. Selected at threshold t iff entry < t.
    """
    weights = probe.get_feature_importance()
    total = float(np.sum(weights))
    order = np.argsort(weights)[::-1]
    prefix = np.concatenate([[0.0], np.cumsum(weights[order])[:-1]]) / total
    return order, prefix  # order[r] = neuron index at rank r, prefix[r] = its entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--pooling_type", type=str, default="mlpneuron_mean")
    parser.add_argument("--suffix", type=str, default="",
                        help="Training-run suffix in the pkl filename, e.g. "
                             "'-std-mlpneuron_mean'.")
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[2500, 5000, 10000],
                        help="Exact total (layer, neuron) pair counts to export.")
    parser.add_argument("--probes_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "train", "probes"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    layers, probes = load_probes(args.model, args.probes_dir,
                                 args.pooling_type, args.suffix)

    # One flat list of (entry_threshold, layer, rank, neuron_idx) over all layers.
    pairs = []
    for layer_idx in layers:
        order, prefix = entry_thresholds(probes[layer_idx])
        pairs.extend((float(prefix[r]), layer_idx, r, int(order[r]))
                     for r in range(len(order)))
    # Sort by entry threshold; tie-break by per-layer rank so within-layer
    # ordering is always by importance.
    pairs.sort(key=lambda p: (p[0], p[2]))
    max_target = max(args.targets)
    if max_target > len(pairs):
        raise SystemExit(f"Only {len(pairs)} (layer, neuron) pairs exist; "
                         f"cannot export {max_target}.")

    for n in sorted(args.targets):
        chosen = pairs[:n]
        eff_t = chosen[-1][0]
        per_layer = {l: [] for l in layers}
        for _, layer_idx, rank, neuron_idx in chosen:
            per_layer[layer_idx].append((rank, neuron_idx))

        out = {f"layer{l}": [idx for _, idx in sorted(per_layer[l])]
               for l in layers if per_layer[l]}
        path = os.path.join(
            args.output_dir,
            f"{args.model}_{args.pooling_type}{args.suffix}"
            f"_selected_neurons_top{n}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        counts = {l: len(v) for l, v in per_layer.items() if v}
        top5 = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        print(f"top{n}: effective threshold ~{eff_t:.4f}, "
              f"{len(counts)}/{len(layers)} layers used, "
              f"biggest layers: {top5}")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()

"""Does shared neuron membership explain a pair's high CKA?

Motivation
----------
The methods overlap unevenly (in an earlier export SIREN and Wang shared 26.5%
of their neurons while SIREN and Zhao shared 4.6%). Shared neurons are
*literally identical columns* in both matrices, which raises CKA for reasons
that have nothing to do with the two methods independently converging on the
same structure.

So for any pair that scores above its null, the question is whether the signal
survives deleting the intersection. This script recomputes CKA on the
disjoint remainders:

    A' = A \\ (A n B)        B' = B \\ (A n B)

with a layer-matched random null built at the SAME reduced per-layer counts,
so the shrunken set size cannot itself explain a change.

Usage:
    python cka/check_overlap_effect.py \
        --activations cka/activations/wildguard_mean.npy \
        --variant class+length
"""
import argparse
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cka_core as core
import neuron_sets as ns
from run_cka import build_design, load_activations, make_rep

HERE = os.path.dirname(os.path.abspath(__file__))


def difference(sel_a, sel_b):
    """sel_a with every (layer, neuron) that also appears in sel_b removed."""
    out = {}
    for layer, idx in sel_a.items():
        other = sel_b.get(layer)
        keep = idx if other is None else np.setdiff1d(idx, other)
        if keep.size:
            out[layer] = keep
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", required=True)
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS))
    parser.add_argument("--budget", type=int, default=ns.DEFAULT_BUDGET,
                        choices=ns.BUDGETS)
    parser.add_argument("--variant", default="class+length",
                        choices=["raw", "class", "class+length"])
    parser.add_argument("--null_seeds", type=int, default=10)
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    methods = list(ns.ALL_METHODS) if args.methods == ["all"] else args.methods
    label = args.label or os.path.splitext(os.path.basename(args.activations))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    zscore = not args.no_zscore

    acts, meta = load_activations(args.activations)
    _, n_layers, width = acts.shape
    Z = build_design(args.variant, meta)
    selections = {m: ns.load_selection(m, args.budget) for m in methods}

    print(f"{acts.shape[0]} prompts | N={args.budget} | variant={args.variant}")
    print("Recomputing each pair after deleting the shared neurons.\n")

    rng_master = np.random.default_rng(args.seed)
    rows = []
    for a, b in itertools.combinations(methods, 2):
        full_a, full_b = selections[a], selections[b]
        cut_a, cut_b = difference(full_a, full_b), difference(full_b, full_a)
        shared = ns.size(full_a) - ns.size(cut_a)

        obs_full = make_rep(acts, full_a, zscore, Z).cka(
            make_rep(acts, full_b, zscore, Z))
        obs_cut = make_rep(acts, cut_a, zscore, Z).cka(
            make_rep(acts, cut_b, zscore, Z))

        null_full, null_cut = [], []
        for _ in range(args.null_seeds):
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            null_full.append(
                make_rep(acts, ns.random_layer_matched(full_a, rng, width), zscore, Z)
                .cka(make_rep(acts, ns.random_layer_matched(full_b, rng, width), zscore, Z)))
            # Null matched to the REDUCED counts, so shrinking the sets is not
            # what moves the number.
            null_cut.append(
                make_rep(acts, ns.random_layer_matched(cut_a, rng, width), zscore, Z)
                .cka(make_rep(acts, ns.random_layer_matched(cut_b, rng, width), zscore, Z)))

        row = {
            "pair": f"{ns.display_name(a)} / {ns.display_name(b)}",
            "shared_neurons": shared,
            "shared_pct": 100.0 * shared / ns.size(full_a),
            "cka_full": obs_full,
            "null_full": float(np.mean(null_full)),
            "z_full": core.null_zscore(obs_full, null_full),
            "cka_disjoint": obs_cut,
            "null_disjoint": float(np.mean(null_cut)),
            "z_disjoint": core.null_zscore(obs_cut, null_cut),
        }
        row["z_change"] = row["z_disjoint"] - row["z_full"]
        rows.append(row)
        print(f"  {row['pair']:34s} shared={shared:4d} ({row['shared_pct']:4.1f}%)  "
              f"z: {row['z_full']:+8.1f} -> {row['z_disjoint']:+8.1f}")

    frame = pd.DataFrame(rows)
    path = os.path.join(
        args.output_dir,
        f"overlap_effect_{label}_N{args.budget}_"
        f"{args.variant.replace('+', '-')}.csv")
    frame.to_csv(path, index=False)
    print(f"\nSaved {path}")

    print("\nInterpretation: a pair whose z stays positive after the shared "
          "neurons are\ndeleted has genuinely convergent populations. A pair "
          "whose z collapses toward\nor below zero was scoring high mainly "
          "because the two matrices contained the\nsame columns.")


if __name__ == "__main__":
    main()

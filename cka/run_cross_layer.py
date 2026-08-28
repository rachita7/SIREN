"""Cross-layer CKA: does method A's layer l population encode what method B's
layer m population encodes, for every (l, m)?

This is the analysis that can actually explain a near-zero neuron overlap
combined with a high aggregate CKA: the methods might localize safety to
different DEPTHS while the populations carry the same structure at different
stages of the network.

The essential correction to the naive version
---------------------------------------------
A raw 32x32 CKA heatmap is close to uninterpretable. Layer l and layer m of a
residual network are intrinsically correlated -- consecutive layers add small
increments to a shared stream -- so ANY neuron subset of layer l scores high
against ANY neuron subset of a nearby layer m. The raw map therefore shows a
broad diagonal band whatever the methods did, and an off-diagonal peak like
"SIREN layer 7 aligns with Yang layer 25" is far more likely to be the model's
own layer geometry than a fact about the selections.

So the same 32x32 map is recomputed with layer-matched RANDOM neurons (same
per-layer counts, same layer pairs) and the reported quantity is

    z(l, m) = ( CKA_selected(l, m) - mean CKA_random(l, m) ) / std CKA_random(l, m)

which is the excess similarity attributable to WHICH neurons were selected,
with the model's layer-to-layer structure divided out.

Sparse cells are masked. Yang's RMS ranking, for instance, puts 1641 of its
2500 neurons in layer 31 and only 2-4 in layers 14-22; CKA computed from 2
neurons is noise, so `--min_neurons` (default 10) blanks those cells.

Usage:
    python cka/run_cross_layer.py \
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
import plots
from run_cka import build_design, load_activations

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")


def per_layer_matrices(acts, sel, layers, zscore, Z, min_neurons):
    """{layer: prepared [N, k_l] matrix} for layers with enough neurons."""
    out = {}
    for layer in layers:
        idx = sel.get(layer)
        if idx is None or len(idx) < min_neurons:
            continue
        X = np.asarray(acts[:, layer, idx], dtype=np.float32)
        try:
            out[layer] = core.prepare(X, zscore=zscore, Z=Z)
        except ValueError:
            continue
    return out


def cross_layer_cka(mats_a, mats_b, num_layers):
    """[num_layers, num_layers] CKA, NaN where either side was masked out.

    Self-norms are cached because the same ||X^T X||_F is reused across the
    whole row/column -- this is what keeps a 32x32 sweep with several random
    draws affordable.
    """
    self_a = {l: np.sqrt(core._fro_sq_cross(X, X)) for l, X in mats_a.items()}
    self_b = {l: np.sqrt(core._fro_sq_cross(X, X)) for l, X in mats_b.items()}
    matrix = np.full((num_layers, num_layers), np.nan)
    for la, Xa in mats_a.items():
        for lb, Xb in mats_b.items():
            den = self_a[la] * self_b[lb]
            if den > 0:
                matrix[la, lb] = core._fro_sq_cross(Xa, Xb) / den
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", required=True)
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS))
    parser.add_argument("--budget", type=int, default=2500, choices=ns.BUDGETS)
    parser.add_argument("--variant", default="class+length",
                        choices=["raw", "class", "class+length"],
                        help="Residualization; the default is the strictest.")
    parser.add_argument("--null_seeds", type=int, default=5,
                        help="Random draws per layer pair. 5 is enough for a "
                             "usable z-score and keeps the sweep to minutes; "
                             "raise it for figures you intend to publish.")
    parser.add_argument("--min_neurons", type=int, default=10,
                        help="Blank cells where either method has fewer "
                             "neurons than this in the layer.")
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument("--include_self_pairs", action="store_true",
                        help="Also emit A-vs-A maps, which visualize the "
                             "model's intrinsic layer-similarity band.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    methods = list(ns.ALL_METHODS) if args.methods == ["all"] else args.methods
    label = args.label or os.path.splitext(os.path.basename(args.activations))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    zscore = not args.no_zscore

    acts, meta = load_activations(args.activations)
    n_prompts, num_layers, width = acts.shape
    layers = list(range(num_layers))
    Z = build_design(args.variant, meta)
    print(f"{n_prompts} prompts | {num_layers} layers | variant={args.variant} "
          f"| min_neurons={args.min_neurons}")

    selections = {m: ns.load_selection(m, args.budget) for m in methods}
    for m in methods:
        counts = ns.layer_counts(selections[m])
        usable = sum(1 for v in counts.values() if v >= args.min_neurons)
        print(f"  {ns.display_name(m):16s} {usable}/{num_layers} layers have "
              f">= {args.min_neurons} neurons")

    print("\nPreparing per-layer matrices for the observed selections ...")
    observed_mats = {m: per_layer_matrices(acts, selections[m], layers, zscore,
                                           Z, args.min_neurons)
                     for m in methods}

    print(f"Preparing {args.null_seeds} layer-matched random draws ...")
    rng_master = np.random.default_rng(args.seed)
    null_mats = []
    for s in range(args.null_seeds):
        rng = np.random.default_rng(rng_master.integers(1 << 62))
        null_mats.append({
            m: per_layer_matrices(
                acts, ns.random_layer_matched(selections[m], rng, width),
                layers, zscore, Z, args.min_neurons)
            for m in methods})
        print(f"  draw {s + 1}/{args.null_seeds}", end="\r", flush=True)
    print(" " * 40, end="\r")

    pairs = list(itertools.combinations(methods, 2))
    if args.include_self_pairs:
        pairs += [(m, m) for m in methods]

    long_rows = []
    for a, b in pairs:
        print(f"\n{ns.display_name(a)} vs {ns.display_name(b)}")
        obs = cross_layer_cka(observed_mats[a], observed_mats[b], num_layers)
        nulls = np.stack([cross_layer_cka(d[a], d[b], num_layers)
                          for d in null_mats])
        null_mean = np.nanmean(nulls, axis=0)
        null_std = np.nanstd(nulls, axis=0, ddof=1) if args.null_seeds > 1 \
            else np.zeros_like(null_mean)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (obs - null_mean) / np.where(null_std > 1e-9, null_std, np.nan)
        mask = np.isfinite(obs) & np.isfinite(z)

        vtag = args.variant.replace("+", "-")
        stem = f"crosslayer_{label}_N{args.budget}_{vtag}_{a}_vs_{b}"
        pd.DataFrame(obs, index=layers, columns=layers).to_csv(
            os.path.join(args.output_dir, stem + "_cka.csv"))
        pd.DataFrame(z, index=layers, columns=layers).to_csv(
            os.path.join(args.output_dir, stem + "_zscore.csv"))
        plots.cross_layer_panel(obs, z, ns.display_name(a), ns.display_name(b),
                                os.path.join(args.output_dir, stem + ".png"),
                                mask=mask)

        for la in layers:
            for lb in layers:
                if not mask[la, lb]:
                    continue
                long_rows.append({
                    "method_a": a, "method_b": b,
                    "layer_a": la, "layer_b": lb,
                    "cka": obs[la, lb], "null_mean": null_mean[la, lb],
                    "null_std": null_std[la, lb], "z": z[la, lb],
                })

        if mask.any():
            flat = np.where(mask, z, -np.inf)
            order = np.dstack(np.unravel_index(
                np.argsort(flat.ravel())[::-1], flat.shape))[0][:5]
            print("  strongest selection-specific layer pairs (by z):")
            for la, lb in order:
                if not np.isfinite(flat[la, lb]) or flat[la, lb] == -np.inf:
                    continue
                print(f"    {ns.display_name(a)} L{la:02d} <-> "
                      f"{ns.display_name(b)} L{lb:02d}  "
                      f"CKA={obs[la, lb]:.4f} null={null_mean[la, lb]:.4f} "
                      f"z={z[la, lb]:+.2f}")
            diag = np.array([z[l, l] if mask[l, l] else np.nan
                             for l in layers], dtype=float)
            if np.isfinite(diag).any():
                print(f"  same-layer z: mean={np.nanmean(diag):+.2f} "
                      f"max={np.nanmax(diag):+.2f} at layer "
                      f"{int(np.nanargmax(diag))}")

    out_csv = os.path.join(
        args.output_dir,
        f"crosslayer_long_{label}_N{args.budget}_{args.variant.replace('+', '-')}.csv")
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print("\nRead the z panel, not the raw CKA panel. Raw CKA is high near the "
          "diagonal\nfor any neuron subset; z > 0 means these particular "
          "selections align more than\nrandom neurons from the same two layers "
          "do. Off-diagonal z peaks are the\ninteresting finding: the same "
          "structure recovered at different network depths.")


if __name__ == "__main__":
    main()

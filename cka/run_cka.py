"""Headline analysis: do the four methods' neuron populations encode the same
representational geometry, even though they select almost disjoint neurons?

For each pair of methods (A, B) this reports linear CKA between
X_A in R^{N x k_A} and X_B in R^{N x k_B} on a held-out prompt set, together
with the three reference quantities that make the number mean anything:

  null (layer-matched random)
      Replace each method's neurons with random neurons having the SAME
      per-layer counts. All four methods are subsets of one shared
      458,752-neuron population, so two arbitrary subsets already inherit the
      model's global variance structure and score a substantial CKA. This is
      the floor. Without it a raw CKA of 0.8 is not evidence of anything.

  null (global random)
      Same total size but uniform over all layers. Comparing it with the
      layer-matched null isolates how much similarity is explained by layer
      placement alone.

  ceiling (same-method disjoint halves)
      CKA between two disjoint halves of ONE method's own selection. This is
      what "the same information, recovered by the same procedure" scores on
      this data -- always below 1.0. Cross-method CKA should be read against
      this, not against 1.0.

and in three residualization variants:

  raw           nothing removed
  class         per-class means removed. A high raw CKA that collapses here
                means the methods only agree on the coarse harmful-vs-benign
                axis.
  class+length  per-class means AND a polynomial in the prompt's token count
                removed. Mean pooling divides by token count, so length leaks
                into every neuron and is frequently a leading principal
                component; harmful and benign prompts differ systematically in
                length. Surviving this variant is the strongest result
                available here: it means the populations agree on fine-grained
                structure WITHIN each class that is not just prompt length.

Two similarity measures are reported side by side. Linear CKA is the standard
choice but is known to be dominated by a few very-high-variance directions
(Davari et al. 2022) -- a real hazard on Llama-3, which has massive-activation
outlier neurons. Spearman RSA on the prompt-similarity matrices is rank-based
and immune to that, so agreement between the two is a meaningful robustness
check.

Usage:
    python cka/run_cka.py --activations cka/activations/wildguard_mean.npy
"""
import argparse
import itertools
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cka_core as core
import neuron_sets as ns
import plots

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")

VARIANTS = ("raw", "class", "class+length")


def load_activations(path, in_memory=True):
    meta_path = os.path.splitext(path)[0] + ".meta.csv"
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing metadata file {meta_path}")
    acts = np.load(path, mmap_mode=None if in_memory else "r")
    meta = pd.read_csv(meta_path)
    if len(meta) != acts.shape[0]:
        raise SystemExit(f"{path} has {acts.shape[0]} rows but "
                         f"{meta_path} has {len(meta)}")
    return acts, meta


def dataset_dummies(meta):
    """One-hot columns for dataset identity, or None for a single source.

    Pooling several corpora into one prompt set introduces a confound of the
    same kind as prompt length: sources differ in style, formatting and topic
    mix, so "which dataset is this prompt from" becomes a strong axis of the
    representation that every method shares regardless of neuron choice.
    Projecting it out keeps the residualized variants honest.
    """
    if "dataset" not in meta.columns:
        return None
    names = sorted(meta["dataset"].astype(str).unique())
    if len(names) < 2:
        return None
    values = meta["dataset"].astype(str).to_numpy()
    # Drop one level; the design matrix already carries an intercept.
    return np.column_stack([(values == n).astype(np.float64) for n in names[1:]])


def build_design(variant, meta):
    if variant == "raw":
        return None
    labels = meta["label"].to_numpy()
    extra = dataset_dummies(meta)
    if variant == "class":
        return core.design_matrix(labels=labels, extra=extra)
    if variant == "class+length":
        return core.design_matrix(labels=labels,
                                  token_counts=meta["n_tokens"].to_numpy(),
                                  extra=extra)
    raise ValueError(variant)


def prepared(acts, sel, zscore, Z):
    X, dropped = core.prepare(ns.build_matrix(acts, sel), zscore=zscore, Z=Z,
                              return_dropped=True)
    return core.Representation(X), dropped


def make_rep(acts, sel, zscore, Z):
    return core.Representation(
        core.prepare(ns.build_matrix(acts, sel), zscore=zscore, Z=Z))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", required=True,
                        help="cka/activations/{tag}_{pooling}.npy from "
                             "cka/extract_activations.py")
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS),
                        help=f"choose from {list(ns.ALL_METHODS)}, or 'all'")
    parser.add_argument("--budget", type=int, default=2500, choices=ns.BUDGETS,
                        help="Neuron budget; all methods are compared at the "
                             "same budget so set sizes cannot drive the result.")
    parser.add_argument("--null_seeds", type=int, default=20,
                        help="Random control draws. 20 gives a usable z-score.")
    parser.add_argument("--ceiling_seeds", type=int, default=10,
                        help="Disjoint-half splits per method.")
    parser.add_argument("--no_zscore", action="store_true",
                        help="Skip per-neuron z-scoring. Reported as a "
                             "sensitivity check only -- without z-scoring a "
                             "single outlier neuron can set the CKA.")
    parser.add_argument("--skip_rsa", action="store_true",
                        help="RSA needs an N x N similarity matrix; skip it if "
                             "memory is tight.")
    parser.add_argument("--rsa_null_seeds", type=int, default=5,
                        help="RSA is the slow measure, so its null uses only "
                             "the first few draws.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None,
                        help="Output filename tag; defaults to the activation "
                             "file's basename.")
    args = parser.parse_args()

    methods = list(ns.ALL_METHODS) if args.methods == ["all"] else args.methods
    for m in methods:
        if m not in ns.METHOD_SPECS:
            raise SystemExit(f"unknown method '{m}'; "
                             f"known: {list(ns.ALL_METHODS)}")
    label = args.label or os.path.splitext(os.path.basename(args.activations))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    zscore = not args.no_zscore

    acts, meta = load_activations(args.activations)
    n_prompts, n_layers, width = acts.shape
    print(f"Activations: {n_prompts} prompts x {n_layers} layers x {width} neurons")
    print(f"  classes: {meta['label'].value_counts().to_dict()}")
    print(f"  token counts by class: "
          f"{meta.groupby('label')['n_tokens'].mean().round(1).to_dict()}")
    if "dataset" in meta.columns and meta["dataset"].nunique() > 1:
        print(f"  pooled sources: {meta['dataset'].value_counts().to_dict()}")
        print(f"  -> dataset identity is projected out in the residualized "
              f"variants alongside class and length")
    if n_prompts < args.budget:
        print(f"  NOTE: {n_prompts} prompts < {args.budget} neurons, so each "
              f"representation is rank-limited by the prompt count. The "
              f"comparison stays valid (all methods are equally limited) but "
              f"absolute CKA values are inflated; rely on the normalized "
              f"score against the controls.")

    selections = {}
    for m in methods:
        selections[m] = ns.load_selection(m, args.budget)
        print("  " + ns.describe(m, args.budget))

    print("\nIndex-level Jaccard (the overlap CKA is meant to look past):")
    jaccard_rows = []
    for a, b in itertools.combinations(methods, 2):
        j = ns.jaccard(selections[a], selections[b])
        jaccard_rows.append({"method_a": a, "method_b": b, "jaccard": j})
        print(f"  {ns.display_name(a):16s} vs {ns.display_name(b):16s} {j:.5f}")

    plots.layer_profiles(
        {ns.display_name(m): ns.layer_counts(selections[m]) for m in methods},
        os.path.join(args.output_dir, f"layer_profiles_{label}_N{args.budget}.png"),
        f"Layer distribution of selected neurons (N={args.budget})")

    rng_master = np.random.default_rng(args.seed)
    all_rows = []
    summary = {
        "activations": os.path.abspath(args.activations),
        "label": label, "budget": args.budget, "methods": methods,
        "num_prompts": int(n_prompts), "zscore": zscore,
        "null_seeds": args.null_seeds, "ceiling_seeds": args.ceiling_seeds,
        "jaccard": jaccard_rows, "variants": {},
    }

    for variant in args.variants:
        print(f"\n{'=' * 72}\nVariant: {variant}\n{'=' * 72}")
        Z = build_design(variant, meta)

        X = {}
        for m in methods:
            X[m], dropped = prepared(acts, selections[m], zscore, Z)
            print(f"  {ns.display_name(m):16s} X shape {X[m].X.shape}"
                  + (f"  ({dropped} near-constant neurons dropped)" if dropped else ""))

        # ------------------------------------------------ ceiling per method
        print("\n  Ceiling: CKA between disjoint halves of the same method")
        ceiling = {}
        for m in methods:
            values = []
            for s in range(args.ceiling_seeds):
                rng = np.random.default_rng(rng_master.integers(1 << 62))
                h1, h2 = ns.split_halves(selections[m], rng)
                A = make_rep(acts, h1, zscore, Z)
                B = make_rep(acts, h2, zscore, Z)
                values.append(A.cka(B))
            ceiling[m] = {"mean": float(np.mean(values)),
                          "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            print(f"    {ns.display_name(m):16s} {ceiling[m]['mean']:.4f} "
                  f"+/- {ceiling[m]['std']:.4f}")

        # ------------------------------------------------------------ nulls
        # One seed at a time: holding every null draw's prepared matrices would
        # cost num_methods * 2 * null_seeds * N * k * 4 bytes (several GB at
        # N=2000, k=2500). Accumulating scalars per pair instead keeps peak
        # memory at two draws' worth.
        print(f"\n  Null: {args.null_seeds} draws of layer-matched and global "
              f"random neuron sets")
        pairs = list(itertools.combinations(methods, 2))
        null_vals = {p: {"layer_matched": [], "global": [], "rsa": []}
                     for p in pairs}
        for s in range(args.null_seeds):
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            lm, gl = {}, {}
            for m in methods:
                lm[m] = make_rep(
                    acts, ns.random_layer_matched(selections[m], rng, width),
                    zscore, Z)
                gl[m] = make_rep(
                    acts, ns.random_global(selections[m], rng, n_layers, width),
                    zscore, Z)
            for p in pairs:
                a, b = p
                null_vals[p]["layer_matched"].append(lm[a].cka(lm[b]))
                null_vals[p]["global"].append(gl[a].cka(gl[b]))
                if not args.skip_rsa and s < args.rsa_null_seeds:
                    null_vals[p]["rsa"].append(
                        core.rsa_spearman(lm[a].X, lm[b].X, seed=args.seed))
            del lm, gl
            print(f"    draw {s + 1}/{args.null_seeds}", end="\r", flush=True)
        print(" " * 40, end="\r")

        # --------------------------------------------------- pairwise values
        rows = []
        for a, b in pairs:
            obs = X[a].cka(X[b])
            obs_unb = X[a].cka_unbiased(X[b])
            obs_rsa = (float("nan") if args.skip_rsa
                       else core.rsa_spearman(X[a].X, X[b].X, seed=args.seed))

            lm_vals = null_vals[(a, b)]["layer_matched"]
            gl_vals = null_vals[(a, b)]["global"]
            lm_rsa = null_vals[(a, b)]["rsa"]

            ceil_mean = float(np.mean([ceiling[a]["mean"], ceiling[b]["mean"]]))
            same_family = (ns.METHOD_SPECS[a]["family"]
                           == ns.METHOD_SPECS[b]["family"])
            row = {
                "variant": variant,
                "method_a": a, "method_b": b,
                "pair": f"{ns.display_name(a)} / {ns.display_name(b)}",
                # Two variants of the same method (e.g. Wang vs Wang-robust)
                # give a second, stricter ceiling than disjoint halves: real
                # procedural differences, same underlying method. A cross-family
                # score at or above the within-family level is a strong result.
                "family_pair": "within" if same_family else "cross",
                "jaccard": ns.jaccard(selections[a], selections[b]),
                "cka": obs,
                "cka_unbiased": obs_unb,
                "rsa_spearman": obs_rsa,
                "null_layer_matched_mean": float(np.mean(lm_vals)),
                "null_layer_matched_std": float(np.std(lm_vals, ddof=1)) if len(lm_vals) > 1 else 0.0,
                "null_global_mean": float(np.mean(gl_vals)),
                "null_global_std": float(np.std(gl_vals, ddof=1)) if len(gl_vals) > 1 else 0.0,
                "ceiling_mean": ceil_mean,
                "z_vs_null": core.null_zscore(obs, lm_vals),
                "normalized": core.normalized_score(obs, float(np.mean(lm_vals)),
                                                    ceil_mean),
                "rsa_null_mean": float(np.mean(lm_rsa)) if lm_rsa else float("nan"),
            }
            rows.append(row)
            print(f"  [{row['family_pair']:6s}] {row['pair']:34s} CKA={obs:.4f}  "
                  f"null={row['null_layer_matched_mean']:.4f}"
                  f"+/-{row['null_layer_matched_std']:.4f}  "
                  f"ceil={ceil_mean:.4f}  z={row['z_vs_null']:+.1f}  "
                  f"norm={row['normalized']:+.3f}")

        all_rows.extend(rows)
        summary["variants"][variant] = {
            "ceiling": ceiling,
            "pairs": rows,
        }

        within = [r["normalized"] for r in rows if r["family_pair"] == "within"]
        cross = [r["normalized"] for r in rows if r["family_pair"] == "cross"]
        if within and cross:
            print(f"\n  mean normalized score: within-method variants "
                  f"{np.mean(within):+.3f}  vs  across methods "
                  f"{np.mean(cross):+.3f}")
            summary["variants"][variant]["mean_normalized"] = {
                "within_family": float(np.mean(within)),
                "cross_family": float(np.mean(cross)),
            }

        # ------------------------------------------------------------- plots
        m_idx = {m: i for i, m in enumerate(methods)}
        k = len(methods)
        obs_mat = np.full((k, k), np.nan)
        null_mat = np.full((k, k), np.nan)
        norm_mat = np.full((k, k), np.nan)
        z_mat = np.full((k, k), np.nan)
        for m in methods:
            obs_mat[m_idx[m], m_idx[m]] = ceiling[m]["mean"]
            norm_mat[m_idx[m], m_idx[m]] = 1.0
        for r in rows:
            i, j = m_idx[r["method_a"]], m_idx[r["method_b"]]
            for mat, key in ((obs_mat, "cka"),
                             (null_mat, "null_layer_matched_mean"),
                             (norm_mat, "normalized"), (z_mat, "z_vs_null")):
                mat[i, j] = mat[j, i] = r[key]

        vtag = variant.replace("+", "-")
        plots.method_matrix_panel(
            [obs_mat, null_mat, norm_mat],
            [ns.display_name(m) for m in methods],
            ["Observed linear CKA\n(diagonal = same-method ceiling)",
             "Layer-matched random neurons\n(the floor)",
             "Normalized score\n0 = random, 1 = same-method ceiling"],
            ["viridis", "viridis", "RdBu_r"],
            os.path.join(args.output_dir,
                         f"cka_matrix_{label}_N{args.budget}_{vtag}.png"),
            suptitle=f"{label} | N={args.budget} | residualization: {variant}",
            vranges=[(None, None), (None, None), (-1.0, 1.5)])

        plots.pair_bars(
            [{"pair": r["pair"], "observed": r["cka"],
              "null_mean": r["null_layer_matched_mean"],
              "null_std": r["null_layer_matched_std"]} for r in rows],
            os.path.join(args.output_dir,
                         f"cka_pairs_{label}_N{args.budget}_{vtag}.png"),
            f"Cross-method CKA vs controls | {label} | {variant}",
            ceiling_by_pair={r["pair"]: r["ceiling_mean"] for r in rows})

    # -------------------------------------------------------------- outputs
    frame = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, f"cka_{label}_N{args.budget}.csv")
    frame.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    json_path = os.path.join(args.output_dir, f"cka_{label}_N{args.budget}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    if len(args.variants) > 1:
        plots.variant_bars(
            {v: [(r["pair"], r["normalized"])
                 for r in all_rows if r["variant"] == v] for v in args.variants},
            os.path.join(args.output_dir,
                         f"cka_variants_{label}_N{args.budget}.png"),
            f"Normalized cross-method similarity by residualization | {label}")

    # ------------------------------------------------------------ reading aid
    print("\n" + "=" * 72)
    print("How to read this")
    print("=" * 72)
    print("normalized ~ 0   : the selected neurons are no more similar to each "
          "other than\n                   random neurons from the same layers. "
          "The methods really do\n                   find different things.")
    print("normalized ~ 1   : as similar as two disjoint halves of one "
          "method's own\n                   selection -- the populations are "
          "representationally\n                   interchangeable despite "
          "near-zero neuron overlap.")
    print("high on 'raw' but low on 'class' : the agreement is only the coarse "
          "harmful-vs-benign axis.")
    print("still high on 'class+length'     : the agreement includes "
          "fine-grained within-class\n                                   "
          "structure. This is the strong result.")
    print("\nAlso check that cka and cka_unbiased agree (prompt-count bias) and "
          "that\nrsa_spearman moves with cka (no single outlier neuron driving "
          "it).")


if __name__ == "__main__":
    main()

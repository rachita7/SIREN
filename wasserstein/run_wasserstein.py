"""Wasserstein comparison of the safety-neuron selections, with the same
controls as cka/run_cka.py so the two measures can be compared row by row.

For each pair of methods (A, B) on held-out prompts this reports

  wasserstein        exact OT distance between the two neuron populations,
                     each neuron a point in prompt space (see wd_core.py).
  matched_corr       1 - wasserstein: mean |corr| between each of A's neurons
                     and its optimally matched partner in B. This is the
                     similarity-oriented reading, and the one all controls
                     are expressed in so that signs read like CKA.
  cka                linear CKA of the SAME prepared matrices, computed here
                     so that CKA and W share identical inputs, seeds, null
                     draws and ceiling splits. (cka/run_cka.py remains the
                     reference CKA pipeline; compare_to_cka.py can merge with
                     either.)

and, for both measures, the reference quantities from cka/run_cka.py:

  null (layer-matched random)   random neurons with each method's per-layer
                                counts -- the floor. Two random neuron sets
                                of size 2294 on 1500 prompts already achieve a
                                matched |corr| of ~0.09 by chance alone.
  null (global random)          same size, uniform over layers.
  ceiling (disjoint halves)     matching one half of a method's OWN selection
                                to the other half, with a half-size random
                                baseline (halving the pool lowers the
                                best-match level by itself).
  residualization variants      raw / class / class+length, as in CKA.

Plus two things CKA cannot give:

  frac_identical, matched_corr_nonidentical
      Neurons selected by BOTH methods match themselves at cost 0. The
      decomposition says how much of the similarity is that trivial overlap
      and what the genuinely different neurons score on their own.
  layer_w1
      Earth-mover's distance between the two depth histograms (in layers).
      A location measure; CKA's layer-matched null divides it out.

Usage:
    python wasserstein/run_wasserstein.py --layer_only                 # no activations
    python wasserstein/run_wasserstein.py \
        --activations cka/activations/wildguard_mean.npy               # CPU
"""
import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "cka"))
sys.path.append(HERE)

import numpy as np
import pandas as pd

import cka_core as core
import neuron_sets as ns
import plots as cka_plots
from run_cka import build_design, load_activations

import wd_core
import wd_plots

DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")
VARIANTS = ("raw", "class", "class+length")


# ---------------------------------------------------------------- helpers

def prepared_with_ids(acts, sel, zscore, Z):
    """Prepared [N, k'] matrix plus the (layer, neuron) identity of each
    surviving column, in the order cka_core.prepare leaves them."""
    raw = ns.build_matrix(acts, sel)
    keep = wd_core.keep_mask(raw)
    X = core.prepare(raw, zscore=zscore, Z=Z)
    if X.shape[1] != int(keep.sum()):
        raise RuntimeError("keep_mask disagrees with cka_core.prepare")
    layers, idx = ns.flatten(sel)
    ids = np.stack([layers, idx], axis=1)[keep]
    return X, ids


def prepared(acts, sel, zscore, Z):
    return core.prepare(ns.build_matrix(acts, sel), zscore=zscore, Z=Z)


def similarity_pair(XA, XB, signed):
    """(matched_corr, cka) for two prepared matrices -- the two measures on
    identical inputs."""
    wd = wd_core.profile_wasserstein(XA, XB, signed=signed)["matched_corr"]
    return wd, core.linear_cka(XA, XB)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return (float(values.mean()),
            float(values.std(ddof=1)) if values.size > 1 else 0.0)


# ----------------------------------------------------------- layer W1 part

def layer_table(methods, selections):
    rows = []
    for a, b in itertools.combinations(methods, 2):
        same_family = ns.METHOD_SPECS[a]["family"] == ns.METHOD_SPECS[b]["family"]
        rows.append({
            "method_a": a, "method_b": b,
            "pair": f"{ns.display_name(a)} / {ns.display_name(b)}",
            "family_pair": "within" if same_family else "cross",
            "jaccard": ns.jaccard(selections[a], selections[b]),
            "layer_w1": wd_core.layer_w1(ns.layer_counts(selections[a]),
                                         ns.layer_counts(selections[b])),
        })
    return rows


def run_layer_only(args, methods, selections, label):
    print("\nLayer W1 (earth-mover's distance between depth histograms; units = layers)")
    print("  per-method distance from a uniform depth profile:")
    per_method = {}
    for m in methods:
        d = wd_core.layer_w1_to_uniform(ns.layer_counts(selections[m]), ns.NUM_LAYERS)
        per_method[m] = d
        counts = ns.layer_counts(selections[m])
        mean_layer = sum(l * c for l, c in counts.items()) / ns.size(selections[m])
        print(f"    {ns.display_name(m):18s} W1(uniform)={d:5.2f}  "
              f"mean layer={mean_layer:5.1f}  layers used={len(counts)}")

    rows = layer_table(methods, selections)
    print("\n  pairwise:")
    for r in rows:
        print(f"    [{r['family_pair']:6s}] {r['pair']:34s} "
              f"layer W1={r['layer_w1']:6.2f}   Jaccard={r['jaccard']:.3f}")

    frame = pd.DataFrame(rows)
    csv = os.path.join(args.output_dir, f"wd_layer_{label}_N{args.budget}.csv")
    frame.to_csv(csv, index=False)
    print(f"\nSaved {csv}")

    k = len(methods)
    mat = np.zeros((k, k))
    m_idx = {m: i for i, m in enumerate(methods)}
    for r in rows:
        i, j = m_idx[r["method_a"]], m_idx[r["method_b"]]
        mat[i, j] = mat[j, i] = r["layer_w1"]
    wd_plots.layer_w1_heatmap(
        mat, [ns.display_name(m) for m in methods],
        os.path.join(args.output_dir, f"wd_layer_{label}_N{args.budget}.png"),
        f"Depth-profile W1 between selections (N={args.budget})")
    cka_plots.layer_profiles(
        {ns.display_name(m): ns.layer_counts(selections[m]) for m in methods},
        os.path.join(args.output_dir,
                     f"wd_layer_profiles_{label}_N{args.budget}.png"),
        f"Layer distribution of selected neurons (N={args.budget})")
    return rows, per_method


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", default=None,
                        help="cka/activations/{tag}_{pooling}.npy from "
                             "cka/extract_activations.py (not needed with "
                             "--layer_only)")
    parser.add_argument("--layer_only", action="store_true",
                        help="Only the depth-histogram W1; needs no activations.")
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS),
                        help=f"choose from {list(ns.ALL_METHODS)}, or 'all'")
    parser.add_argument("--budget", type=int, default=ns.DEFAULT_BUDGET,
                        choices=ns.BUDGETS)
    parser.add_argument("--null_seeds", type=int, default=20)
    parser.add_argument("--ceiling_seeds", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--signed", action="store_true",
                        help="Cost 1 - rho instead of 1 - |rho|. Then W is "
                             "exactly W2^2/(2N) between the z-scored point "
                             "clouds, but a neuron and its negation count as "
                             "different features (CKA does not distinguish them).")
    parser.add_argument("--strong", type=float, default=0.5,
                        help="|rho| threshold for the frac_strong column.")
    parser.add_argument("--no_zscore", action="store_true",
                        help="Affects only the CKA columns; the correlation "
                             "cost is scale-invariant per neuron by construction.")
    parser.add_argument("--save_matching", action="store_true",
                        help="Write the optimal neuron matching of every "
                             "observed pair to results/matchings/.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None,
                        help="Output tag; defaults to the activation file's "
                             "basename ('selections' with --layer_only).")
    args = parser.parse_args()

    methods = list(ns.ALL_METHODS) if args.methods == ["all"] else args.methods
    for m in methods:
        if m not in ns.METHOD_SPECS:
            raise SystemExit(f"unknown method '{m}'; known: {list(ns.ALL_METHODS)}")
    if len(methods) < 2:
        raise SystemExit("need at least two methods")
    os.makedirs(args.output_dir, exist_ok=True)

    selections = {m: ns.load_selection(m, args.budget) for m in methods}
    print(f"Selections at N={args.budget}:")
    for m in methods:
        print("  " + ns.describe(m, args.budget))

    if args.layer_only:
        label = args.label or "selections"
        run_layer_only(args, methods, selections, label)
        return
    if not args.activations:
        raise SystemExit("--activations is required unless --layer_only")

    label = args.label or os.path.splitext(os.path.basename(args.activations))[0]
    zscore = not args.no_zscore

    acts, meta = load_activations(args.activations)
    n_prompts, n_layers, width = acts.shape
    print(f"\nActivations: {n_prompts} prompts x {n_layers} layers x {width} neurons")
    print(f"  classes: {meta['label'].value_counts().to_dict()}")
    if "dataset" in meta.columns and meta["dataset"].nunique() > 1:
        print(f"  pooled sources: {meta['dataset'].value_counts().to_dict()}")
    if n_prompts < args.budget:
        print(f"  NOTE: {n_prompts} prompts < {args.budget} neurons. Each neuron "
              f"is a point in R^{n_prompts}; with more points than dimensions "
              f"the chance level of the best match rises. The layer-matched "
              f"null accounts for this; read z and normalized, not raw values.")
    chance = wd_core.chance_matched_corr(n_prompts, min(args.budget, 3000))
    print(f"  chance matched |corr| for two independent Gaussian populations "
          f"of this size: ~{chance:.3f}  (real null uses model neurons)")

    layer_rows, layer_uniform = run_layer_only(args, methods, selections, label)
    layer_w1_by_pair = {(r["method_a"], r["method_b"]): r["layer_w1"]
                        for r in layer_rows}

    rng_master = np.random.default_rng(args.seed)
    pairs = list(itertools.combinations(methods, 2))
    all_rows = []
    summary = {
        "activations": os.path.abspath(args.activations), "label": label,
        "budget": args.budget, "methods": methods, "num_prompts": int(n_prompts),
        "signed_cost": args.signed, "zscore": zscore,
        "null_seeds": args.null_seeds, "ceiling_seeds": args.ceiling_seeds,
        "chance_matched_corr_gaussian": chance,
        "layer_w1": layer_rows,
        "layer_w1_to_uniform": layer_uniform,
        "variants": {},
    }
    if args.save_matching:
        os.makedirs(os.path.join(args.output_dir, "matchings"), exist_ok=True)

    for variant in args.variants:
        print(f"\n{'=' * 72}\nVariant: {variant}\n{'=' * 72}")
        Z = build_design(variant, meta)

        X, ids = {}, {}
        for m in methods:
            X[m], ids[m] = prepared_with_ids(acts, selections[m], zscore, Z)
            print(f"  {ns.display_name(m):16s} X shape {X[m].shape}")

        # ------------------------------------------------------- ceiling
        print("\n  Ceiling: disjoint halves of the same method "
              "(with a half-size random baseline)")
        ceiling = {}
        for m in methods:
            wd_vals, wd_half_null, cka_vals, cka_half_null = [], [], [], []
            for _ in range(args.ceiling_seeds):
                rng = np.random.default_rng(rng_master.integers(1 << 62))
                h1, h2 = ns.split_halves(selections[m], rng)
                w, c = similarity_pair(prepared(acts, h1, zscore, Z),
                                       prepared(acts, h2, zscore, Z), args.signed)
                wd_vals.append(w)
                cka_vals.append(c)
                rand = ns.random_layer_matched(selections[m], rng, width)
                r1, r2 = ns.split_halves(rand, rng)
                w, c = similarity_pair(prepared(acts, r1, zscore, Z),
                                       prepared(acts, r2, zscore, Z), args.signed)
                wd_half_null.append(w)
                cka_half_null.append(c)
            ceiling[m] = {}
            for key, vals, nulls in (("wd", wd_vals, wd_half_null),
                                     ("cka", cka_vals, cka_half_null)):
                mean, std = stats(vals)
                ceiling[m][key] = {
                    "mean": mean, "std": std,
                    "null_half_mean": float(np.mean(nulls)),
                    "excess": float(mean - np.mean(nulls)),
                }
            w, c = ceiling[m]["wd"], ceiling[m]["cka"]
            print(f"    {ns.display_name(m):16s} matched|corr| {w['mean']:.3f} "
                  f"(half-null {w['null_half_mean']:.3f}, excess {w['excess']:+.3f})"
                  f"   CKA {c['mean']:.3f} (half-null {c['null_half_mean']:.3f}, "
                  f"excess {c['excess']:+.3f})")

        # --------------------------------------------------------- nulls
        print(f"\n  Null: {args.null_seeds} draws of layer-matched and global "
              f"random neuron sets")
        null = {p: {"wd_lm": [], "wd_gl": [], "cka_lm": [], "cka_gl": []}
                for p in pairs}
        null_rho_sample = []
        for s in range(args.null_seeds):
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            lm, gl = {}, {}
            for m in methods:
                lm[m] = prepared(acts, ns.random_layer_matched(selections[m], rng, width),
                                 zscore, Z)
                gl[m] = prepared(acts, ns.random_global(selections[m], rng, n_layers, width),
                                 zscore, Z)
            for p in pairs:
                a, b = p
                res = wd_core.profile_wasserstein(lm[a], lm[b], signed=args.signed)
                null[p]["wd_lm"].append(res["matched_corr"])
                if s == 0:
                    null_rho_sample.append(res["rho"])
                null[p]["cka_lm"].append(core.linear_cka(lm[a], lm[b]))
                w, c = similarity_pair(gl[a], gl[b], args.signed)
                null[p]["wd_gl"].append(w)
                null[p]["cka_gl"].append(c)
            del lm, gl
            print(f"    draw {s + 1}/{args.null_seeds}", end="\r", flush=True)
        print(" " * 40, end="\r")

        # ------------------------------------------------------ observed
        rows = []
        rho_by_pair = {}
        for a, b in pairs:
            obs = wd_core.profile_wasserstein(X[a], X[b], signed=args.signed,
                                              ids_a=ids[a], ids_b=ids[b],
                                              strong=args.strong)
            cka_obs = core.linear_cka(X[a], X[b])
            pair_name = f"{ns.display_name(a)} / {ns.display_name(b)}"
            rho_by_pair[pair_name] = obs["rho"]
            same_family = ns.METHOD_SPECS[a]["family"] == ns.METHOD_SPECS[b]["family"]

            row = {
                "variant": variant, "method_a": a, "method_b": b,
                "pair": pair_name,
                "family_pair": "within" if same_family else "cross",
                "jaccard": ns.jaccard(selections[a], selections[b]),
                "layer_w1": layer_w1_by_pair[(a, b)],
                # ---- Wasserstein
                "wasserstein": obs["wasserstein"],
                "matched_corr": obs["matched_corr"],
                "matched_corr_median": obs["matched_corr_median"],
                "frac_strong": obs["frac_strong"],
                "frac_identical": obs["frac_identical"],
                "matched_corr_nonidentical": obs["matched_corr_nonidentical"],
                "n_matched": obs["n_matched"], "n_unmatched": obs["n_unmatched"],
                # ---- CKA on the same inputs
                "cka": cka_obs,
            }
            for key, measure_obs in (("wd", obs["matched_corr"]), ("cka", cka_obs)):
                lm_vals = null[(a, b)][f"{key}_lm"]
                gl_vals = null[(a, b)][f"{key}_gl"]
                lm_mean, lm_std = stats(lm_vals)
                gl_mean, gl_std = stats(gl_vals)
                ceil_mean = float(np.mean([ceiling[a][key]["mean"],
                                           ceiling[b][key]["mean"]]))
                ceil_excess = float(np.mean([ceiling[a][key]["excess"],
                                             ceiling[b][key]["excess"]]))
                row.update({
                    f"{key}_null_layer_matched_mean": lm_mean,
                    f"{key}_null_layer_matched_std": lm_std,
                    f"{key}_null_global_mean": gl_mean,
                    f"{key}_null_global_std": gl_std,
                    f"{key}_ceiling_mean": ceil_mean,
                    f"{key}_ceiling_excess": ceil_excess,
                    f"{key}_z_vs_null": core.null_zscore(measure_obs, lm_vals),
                    f"{key}_normalized": core.normalized_score(
                        measure_obs, lm_mean, lm_mean + ceil_excess),
                })
            rows.append(row)

            if args.save_matching:
                pd.DataFrame({
                    "layer_a": ids[a][obs["rows"], 0], "neuron_a": ids[a][obs["rows"], 1],
                    "layer_b": ids[b][obs["cols"], 0], "neuron_b": ids[b][obs["cols"], 1],
                    "rho": obs["rho"], "identical": obs["identical"],
                }).sort_values("rho", key=np.abs, ascending=False).to_csv(
                    os.path.join(args.output_dir, "matchings",
                                 f"matching_{label}_N{args.budget}_"
                                 f"{variant.replace('+', '-')}_{a}_vs_{b}.csv"),
                    index=False)

            def fmt(v):
                return f"{v:+.2f}" if np.isfinite(v) else " n/a"
            print(f"  [{row['family_pair']:6s}] {pair_name:34s} "
                  f"W={row['wasserstein']:.3f} matched|corr|={row['matched_corr']:.3f} "
                  f"(null {row['wd_null_layer_matched_mean']:.3f}, "
                  f"identical {row['frac_identical']:.2f}, "
                  f"non-identical {row['matched_corr_nonidentical']:.3f})  "
                  f"z={row['wd_z_vs_null']:+.1f} norm={fmt(row['wd_normalized'])}"
                  f"   | CKA={cka_obs:.3f} z={row['cka_z_vs_null']:+.1f} "
                  f"norm={fmt(row['cka_normalized'])}")

        all_rows.extend(rows)
        summary["variants"][variant] = {"ceiling": ceiling, "pairs": rows}

        # ------------------------------------------------ agreement summary
        frame = pd.DataFrame(rows)
        ok = frame[["wd_normalized", "cka_normalized"]].notna().all(axis=1)
        if ok.sum() >= 3:
            from scipy.stats import spearmanr
            rho_n, _ = spearmanr(frame.loc[ok, "wd_normalized"],
                                 frame.loc[ok, "cka_normalized"])
            rho_z, _ = spearmanr(frame["wd_z_vs_null"], frame["cka_z_vs_null"])
            summary["variants"][variant]["agreement"] = {
                "spearman_normalized": float(rho_n), "spearman_z": float(rho_z)}
            print(f"\n  CKA vs Wasserstein rank agreement over {len(frame)} pairs: "
                  f"Spearman rho = {rho_n:+.2f} (normalized), {rho_z:+.2f} (z)")
        n_wd = int((frame["wd_z_vs_null"] > 3).sum())
        n_cka = int((frame["cka_z_vs_null"] > 3).sum())
        print(f"  pairs significantly above the layer-matched null (z > 3): "
              f"W {n_wd}/{len(frame)}, CKA {n_cka}/{len(frame)}")
        disagree = frame[(frame["cka_z_vs_null"] > 3) & (frame["wd_z_vs_null"] <= 3)]
        if len(disagree):
            print("  high CKA but chance-level matching (same subspace, "
                  "different individual neurons):")
            for _, r in disagree.iterrows():
                print(f"    {r['pair']}")

        # --------------------------------------------------------- plots
        k = len(methods)
        m_idx = {m: i for i, m in enumerate(methods)}
        obs_mat, null_mat, wd_norm, cka_norm = (np.full((k, k), np.nan) for _ in range(4))
        for m in methods:
            obs_mat[m_idx[m], m_idx[m]] = ceiling[m]["wd"]["mean"]
            wd_norm[m_idx[m], m_idx[m]] = cka_norm[m_idx[m], m_idx[m]] = 1.0
        for r in rows:
            i, j = m_idx[r["method_a"]], m_idx[r["method_b"]]
            for mat, key in ((obs_mat, "matched_corr"),
                             (null_mat, "wd_null_layer_matched_mean"),
                             (wd_norm, "wd_normalized"), (cka_norm, "cka_normalized")):
                mat[i, j] = mat[j, i] = r[key]
        vtag = variant.replace("+", "-")
        labels = [ns.display_name(m) for m in methods]
        cka_plots.method_matrix_panel(
            [obs_mat, null_mat, wd_norm, cka_norm], labels,
            ["Matched |corr| = 1 - W\n(diagonal = same-method ceiling)",
             "Layer-matched random neurons\n(the floor)",
             "Wasserstein normalized\n0 = random, 1 = ceiling",
             "CKA normalized (same inputs)\n0 = random, 1 = ceiling"],
            ["viridis", "viridis", "RdBu_r", "RdBu_r"],
            os.path.join(args.output_dir, f"wd_matrix_{label}_N{args.budget}_{vtag}.png"),
            suptitle=f"{label} | N={args.budget} | residualization: {variant}",
            vranges=[(None, None), (None, None), (-1.0, 1.5), (-1.0, 1.5)])
        wd_plots.pair_bars(
            [{"pair": r["pair"], "observed": r["matched_corr"],
              "null_mean": r["wd_null_layer_matched_mean"],
              "null_std": r["wd_null_layer_matched_std"]} for r in rows],
            os.path.join(args.output_dir, f"wd_pairs_{label}_N{args.budget}_{vtag}.png"),
            f"Matched |corr| (1 - W) vs controls | {label} | {variant}",
            "mean |corr| of optimally matched neurons  (= 1 - W)",
            ceiling_by_pair={r["pair"]: r["wd_ceiling_mean"] for r in rows})
        wd_plots.matched_corr_hist(
            rho_by_pair,
            os.path.join(args.output_dir, f"wd_matchhist_{label}_N{args.budget}_{vtag}.png"),
            f"Per-neuron match quality | {label} | {variant}",
            null_rho=np.concatenate(null_rho_sample) if null_rho_sample else None)
        wd_plots.scatter_measures(
            frame["cka_normalized"], frame["wd_normalized"], frame["pair"],
            os.path.join(args.output_dir, f"wd_vs_cka_{label}_N{args.budget}_{vtag}.png"),
            f"CKA vs Wasserstein, same inputs | {label} | {variant}",
            "CKA normalized (0 = random, 1 = ceiling)",
            "Wasserstein normalized (0 = random, 1 = ceiling)",
            family=frame["family_pair"],
            stats_text=(f"Spearman rho = {rho_n:+.2f}" if ok.sum() >= 3 else None))

    # -------------------------------------------------------------- outputs
    frame = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, f"wd_{label}_N{args.budget}.csv")
    frame.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    json_path = os.path.join(args.output_dir, f"wd_{label}_N{args.budget}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved {json_path}")

    if len(args.variants) > 1:
        cka_plots.variant_bars(
            {v: [(r["pair"], r["wd_normalized"]) for r in all_rows if r["variant"] == v]
             for v in args.variants},
            os.path.join(args.output_dir, f"wd_variants_{label}_N{args.budget}.png"),
            f"Normalized Wasserstein similarity by residualization | {label}")

    print("\n" + "=" * 72 + "\nHow to read this\n" + "=" * 72)
    print("wasserstein      : distance; 0 = the two neuron clouds coincide. Raw values\n"
          "                   sit near 1 - chance level for ANY two sets, so never\n"
          "                   report them without the null.")
    print("matched_corr     : 1 - W. Mean |corr| between each neuron and its optimal\n"
          "                   partner. wd_z_vs_null / wd_normalized are on this scale,\n"
          "                   so signs read like CKA (higher = more similar).")
    print("frac_identical   : share of matches that are the SAME neuron (Jaccard\n"
          "                   leaking in at cost 0). matched_corr_nonidentical is\n"
          "                   the honest number for the disjoint parts.")
    print("CKA high, W low  : same subspace, different individual neurons -- the\n"
          "                   rotation invariance of CKA. Expected to be common.")
    print("Both high        : neurons are interchangeable as individuals. Strong.")
    print("Both ~ 0         : the methods find different things; Jaccard was the story.")
    print("layer_w1         : depth disagreement in layers. Orthogonal to the above.")


if __name__ == "__main__":
    main()

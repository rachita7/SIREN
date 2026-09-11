"""Do CKA and Wasserstein rank the method pairs the same way?

The supervisor's question is not "are the raw numbers close" -- they are on
different scales and measure different things (see wd_core.py). It is whether
the two measures AGREE ON WHICH PAIRS ARE SIMILAR, once each is expressed
relative to its own null and ceiling. This script reports, per residualization
variant:

  Spearman rho between the normalized scores      (rank agreement)
  Spearman rho between the z-scores vs null       (rank agreement, no ceiling)
  Pearson r between the normalized scores
  significance agreement: pairs above the null (z > 3) under both / one / none
  the pairs in the "CKA high, W low" quadrant    (same subspace, different
                                                  individual neurons)

Two sources of CKA numbers are supported:

  internal   run_wasserstein.py computes CKA on the very same prepared
             matrices and null draws it uses for W; those columns
             (cka_normalized, cka_z_vs_null) are in the W table already.
             This is the cleanest apples-to-apples comparison.
  external   --cka cka/results/cka_{tag}_N{budget}.csv from cka/run_cka.py:
             the reference CKA pipeline (unbiased CKA, RSA, its own seeds).
             Rows are merged on (variant, method_a, method_b); the observed
             CKA values should coincide to ~1e-4 since they are deterministic
             given the activations -- that is checked and printed.

Usage:
    python wasserstein/compare_to_cka.py \
        --wasserstein wasserstein/results/wd_wildguard_mean_N2294.csv
    python wasserstein/compare_to_cka.py \
        --wasserstein wasserstein/results/wd_wildguard_mean_N2294.csv \
        --cka cka/results/cka_wildguard_mean_N2294.csv
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import wd_plots


def agreement(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    return (float(spearmanr(x[ok], y[ok])[0]),
            float(pearsonr(x[ok], y[ok])[0]), int(ok.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wasserstein", required=True,
                        help="wd_{label}_N{budget}.csv from run_wasserstein.py")
    parser.add_argument("--cka", default=None,
                        help="cka_{label}_N{budget}.csv from cka/run_cka.py. "
                             "Omitted: use the CKA columns computed inside the "
                             "Wasserstein run (same inputs and seeds).")
    parser.add_argument("--z_threshold", type=float, default=3.0)
    parser.add_argument("--output_dir", default=None,
                        help="Defaults to the directory of --wasserstein.")
    args = parser.parse_args()

    wd = pd.read_csv(args.wasserstein)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.wasserstein))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.wasserstein))[0]
    stem = stem[len("wd_"):] if stem.startswith("wd_") else stem

    if args.cka:
        cka = pd.read_csv(args.cka)
        keep = ["variant", "method_a", "method_b", "cka", "cka_unbiased",
                "rsa_spearman", "z_vs_null", "normalized"]
        keep = [c for c in keep if c in cka.columns]
        cka = cka[keep].rename(columns={
            "cka": "cka_ext", "cka_unbiased": "cka_unbiased_ext",
            "rsa_spearman": "rsa_ext", "z_vs_null": "cka_z_ext",
            "normalized": "cka_norm_ext"})
        merged = wd.merge(cka, on=["variant", "method_a", "method_b"], how="inner")
        # Both files list pairs as itertools.combinations over the methods
        # list; if a user reordered methods, try the swapped orientation too.
        if len(merged) < len(wd):
            swapped = cka.rename(columns={"method_a": "method_b", "method_b": "method_a"})
            more = wd.merge(swapped, on=["variant", "method_a", "method_b"], how="inner")
            merged = pd.concat([merged, more]).drop_duplicates(
                ["variant", "method_a", "method_b"])
        if merged.empty:
            raise SystemExit(
                "no rows in common. Both tables must be run at the same budget "
                "and with the same --methods (the CKA table's rows are "
                f"{sorted(set(cka['method_a']) | set(cka['method_b']))}, the W "
                f"table's are {sorted(set(wd['method_a']) | set(wd['method_b']))}).")
        if len(merged) < len(wd):
            print(f"NOTE: {len(wd) - len(merged)} of {len(wd)} W rows have no CKA "
                  f"counterpart and are dropped from the comparison.")
        if "cka_ext" in merged:
            gap = float((merged["cka"] - merged["cka_ext"]).abs().max())
            print(f"Observed CKA, internal vs cka/run_cka.py: max |diff| = {gap:.2e}"
                  + ("" if gap < 1e-3 else
                     "  <-- differs; check that both runs used the same "
                     "activations and --no_zscore setting"))
        cka_norm_col, cka_z_col, source = "cka_norm_ext", "cka_z_ext", "cka/run_cka.py"
    else:
        merged = wd.copy()
        cka_norm_col, cka_z_col, source = "cka_normalized", "cka_z_vs_null", "internal"

    print(f"\nCKA source: {source}.  Pairs per variant: "
          f"{merged.groupby('variant').size().to_dict()}\n")
    out_rows = []
    for variant, sub in merged.groupby("variant", sort=False):
        rho_n, r_n, n_ok = agreement(sub["wd_normalized"], sub[cka_norm_col])
        rho_z, r_z, _ = agreement(sub["wd_z_vs_null"], sub[cka_z_col])
        rho_raw, _, _ = agreement(sub["matched_corr"], sub["cka"])
        rho_j_wd, _, _ = agreement(sub["jaccard"], sub["wd_z_vs_null"])
        rho_j_cka, _, _ = agreement(sub["jaccard"], sub[cka_z_col])
        rho_layer_wd, _, _ = agreement(-sub["layer_w1"], sub["wd_z_vs_null"])
        rho_layer_cka, _, _ = agreement(-sub["layer_w1"], sub[cka_z_col])

        wd_sig = sub["wd_z_vs_null"] > args.z_threshold
        cka_sig = sub[cka_z_col] > args.z_threshold
        both = int((wd_sig & cka_sig).sum())
        cka_only = int((cka_sig & ~wd_sig).sum())
        wd_only = int((wd_sig & ~cka_sig).sum())
        neither = int((~wd_sig & ~cka_sig).sum())

        print(f"=== {variant} ===")
        print(f"  rank agreement, normalized scores : Spearman {rho_n:+.2f} "
              f"(Pearson {r_n:+.2f}, n={n_ok})")
        print(f"  rank agreement, z vs null         : Spearman {rho_z:+.2f}")
        print(f"  rank agreement, raw values        : Spearman {rho_raw:+.2f}  "
              f"(matched|corr| vs CKA; scale-dependent, for reference)")
        print(f"  significance (z > {args.z_threshold:g}): both {both}, CKA only "
              f"{cka_only}, W only {wd_only}, neither {neither}")
        print(f"  what each measure tracks: Jaccard -> W {rho_j_wd:+.2f}, "
              f"CKA {rho_j_cka:+.2f};  depth proximity -> W {rho_layer_wd:+.2f}, "
              f"CKA {rho_layer_cka:+.2f}")
        if cka_only:
            print("  CKA-significant but chance-level matching (same subspace, "
                  "different individual neurons):")
            for _, r in sub[cka_sig & ~wd_sig].iterrows():
                print(f"    {r['pair']:34s} CKA z={r[cka_z_col]:+.1f}  "
                      f"W z={r['wd_z_vs_null']:+.1f}")
        if wd_only:
            print("  W-significant but not CKA (unusual; check frac_identical -- "
                  "shared neurons match at cost 0):")
            for _, r in sub[wd_sig & ~cka_sig].iterrows():
                print(f"    {r['pair']:34s} W z={r['wd_z_vs_null']:+.1f}  "
                      f"identical {r['frac_identical']:.2f}")
        print()

        out_rows.append({
            "variant": variant, "n_pairs": len(sub), "cka_source": source,
            "spearman_normalized": rho_n, "pearson_normalized": r_n,
            "spearman_z": rho_z, "spearman_raw": rho_raw,
            "sig_both": both, "sig_cka_only": cka_only, "sig_wd_only": wd_only,
            "sig_neither": neither,
            "spearman_jaccard_vs_wd_z": rho_j_wd,
            "spearman_jaccard_vs_cka_z": rho_j_cka,
            "spearman_depth_vs_wd_z": rho_layer_wd,
            "spearman_depth_vs_cka_z": rho_layer_cka,
        })

        vtag = variant.replace("+", "-")
        wd_plots.scatter_measures(
            sub[cka_norm_col], sub["wd_normalized"], sub["pair"],
            os.path.join(out_dir, f"wd_vs_cka_{stem}_{vtag}_{'ext' if args.cka else 'int'}.png"),
            f"CKA ({source}) vs Wasserstein | {stem} | {variant}",
            "CKA normalized (0 = random, 1 = ceiling)",
            "Wasserstein normalized (0 = random, 1 = ceiling)",
            family=sub["family_pair"],
            stats_text=f"Spearman rho = {rho_n:+.2f}\nz-score rho = {rho_z:+.2f}")

    summary = pd.DataFrame(out_rows)
    path = os.path.join(out_dir, f"wd_vs_cka_{stem}.csv")
    summary.to_csv(path, index=False)
    pairs_path = os.path.join(out_dir, f"wd_vs_cka_{stem}_pairs.csv")
    merged.to_csv(pairs_path, index=False)
    print(f"Saved {path}\nSaved {pairs_path}")

    print("\nReading: rho >= 0.7 -- the two measures order the pairs the same way;\n"
          "         0.3-0.7   -- broadly consistent, look at the quadrant lists;\n"
          "         <= 0.3    -- they disagree. That is a finding, not an error:\n"
          "         CKA is invariant to rotations within a population, W is not.\n"
          "         Pairs with high CKA and chance-level W share a subspace but\n"
          "         not individual neurons.")


if __name__ == "__main__":
    main()

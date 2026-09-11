"""Known-answer checks for the Wasserstein measures plus an end-to-end run on
synthetic activations. No GPU, ~30 s.

    python wasserstein/smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "cka"))
sys.path.append(HERE)

import numpy as np
import pandas as pd

import cka_core as core
import neuron_sets as ns

import wd_core


def _load_cka_synthetic():
    """cka/smoke_test.py's synthetic-activation generator. Loaded by path:
    this file is also called smoke_test.py and sits first on sys.path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cka_smoke_test", os.path.join(REPO, "cka", "smoke_test.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_synthetic


cka_make_synthetic = _load_cka_synthetic()


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        check.failures += 1


check.failures = 0


def test_measures():
    print("\nMeasure sanity checks")
    rng = np.random.default_rng(0)
    n, k = 400, 60
    X = core.prepare(rng.normal(size=(n, k)), zscore=True)

    # ---- identity, permutation, sign: W = 0 and CKA = 1 in all three cases
    self_ = wd_core.profile_wasserstein(X, X)
    check("W(X, X) == 0", self_["wasserstein"] < 1e-5, f"W={self_['wasserstein']:.2e}")
    perm = X[:, rng.permutation(k)]
    p = wd_core.profile_wasserstein(X, perm)
    check("W invariant to column permutation (neuron order is arbitrary)",
          p["wasserstein"] < 1e-5, f"W={p['wasserstein']:.2e}")
    signs = rng.choice([-1.0, 1.0], size=k)
    s = wd_core.profile_wasserstein(X, X * signs)
    check("W invariant to per-neuron sign flips (default |rho| cost), like CKA",
          s["wasserstein"] < 1e-5, f"W={s['wasserstein']:.2e}")
    s_signed = wd_core.profile_wasserstein(X, X * signs, signed=True)
    check("--signed cost DOES see sign flips", s_signed["wasserstein"] > 0.05,
          f"W={s_signed['wasserstein']:.3f}")
    check("CKA also invariant to sign flips (so |rho| is the fair default)",
          abs(core.linear_cka(X, core.prepare(X * signs, zscore=False)) - 1) < 1e-6)

    # ---- the defining difference: rotate the basis of the same subspace
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    XQ = core.prepare(X @ Q, zscore=True)
    rot = wd_core.profile_wasserstein(X, XQ)
    cka_rot = core.linear_cka(X, core.prepare(X @ Q, zscore=False))
    check("rotation: CKA stays 1 ...", abs(cka_rot - 1) < 1e-5, f"CKA={cka_rot:.5f}")
    check("... but W moves far from 0 (individual neurons no longer match)",
          rot["wasserstein"] > 0.5, f"W={rot['wasserstein']:.3f}")

    # ---- signed W equals the Euclidean 2-Wasserstein up to the 2N constant
    Y = core.prepare(rng.normal(size=(n, k)), zscore=True)
    sw = wd_core.profile_wasserstein(X, Y, signed=True)
    from scipy.optimize import linear_sum_assignment
    D2 = ((X[:, :, None] - Y[:, None, :]) ** 2).sum(axis=0)          # [k, k]
    r_, c_ = linear_sum_assignment(D2)
    w2_sq = D2[r_, c_].mean()
    check("signed W == W2^2 / (2N) on z-scored clouds",
          abs(sw["wasserstein"] - w2_sq / (2 * n)) < 1e-3,
          f"{sw['wasserstein']:.4f} vs {w2_sq / (2 * n):.4f}")

    # ---- independent populations sit at the chance level, and it is not 0
    ind = wd_core.profile_wasserstein(X, Y)
    check("independent populations: matched |corr| well below 1 but > 0",
          0.02 < ind["matched_corr"] < 0.5,
          f"matched|corr|={ind['matched_corr']:.3f} (this is why the null exists)")
    chance = wd_core.chance_matched_corr(n, k, seed=1)
    check("chance helper agrees with direct computation",
          abs(chance - ind["matched_corr"]) < 0.05,
          f"{chance:.3f} vs {ind['matched_corr']:.3f}")

    # ---- shared latent, different neurons: CKA high, W partial
    latent = rng.normal(size=(n, 5))
    A = core.prepare(latent @ rng.normal(size=(5, 40)) + 0.3 * rng.normal(size=(n, 40)))
    B = core.prepare(latent @ rng.normal(size=(5, 40)) + 0.3 * rng.normal(size=(n, 40)))
    shared = wd_core.profile_wasserstein(A, B)
    check("shared latent: matching finds correlated partners (>> chance)",
          shared["matched_corr"] > ind["matched_corr"] + 0.2,
          f"{shared['matched_corr']:.3f} vs chance {ind['matched_corr']:.3f}")
    check("shared latent: still far from identical (W > 0.1)",
          shared["wasserstein"] > 0.1, f"W={shared['wasserstein']:.3f}")

    # ---- decomposition into identical / non-identical matches
    ids_a = np.stack([np.zeros(k, int), np.arange(k)], axis=1)
    ids_b = ids_a.copy()
    ids_b[k // 2:, 1] += 10_000                       # second half: new neurons
    mixed = np.concatenate([X[:, :k // 2], Y[:, k // 2:]], axis=1)
    d = wd_core.profile_wasserstein(X, mixed, ids_a=ids_a, ids_b=ids_b)
    check("half the neurons shared -> frac_identical == 0.5",
          abs(d["frac_identical"] - 0.5) < 1e-9, f"{d['frac_identical']:.3f}")
    check("non-identical part scores at chance, whole scores in between",
          d["matched_corr_nonidentical"] < 0.5 < d["matched_corr"],
          f"nonidentical={d['matched_corr_nonidentical']:.3f} "
          f"all={d['matched_corr']:.3f}")

    # ---- CKA written from correlation matrices equals cka_core on z-scored data
    R_ab = wd_core.cross_correlation(A, B)
    c_from_R = wd_core.cka_from_correlations(R_ab, wd_core.cross_correlation(A, A),
                                             wd_core.cross_correlation(B, B))
    check("CKA from the correlation matrix == cka_core.linear_cka",
          abs(c_from_R - core.linear_cka(A, B)) < 1e-5,
          f"{c_from_R:.5f} vs {core.linear_cka(A, B):.5f}")

    # ---- rectangular (unequal sizes) handled: min(k) matched, rest unmatched
    rect = wd_core.profile_wasserstein(X, Y[:, :k - 7])
    check("unequal sizes: n_matched = min(k), n_unmatched = difference",
          rect["n_matched"] == k - 7 and rect["n_unmatched"] == 7)
    check("W symmetric", abs(wd_core.profile_wasserstein(Y, X)["wasserstein"]
                             - ind["wasserstein"]) < 1e-9)

    # ---- layer W1
    check("layer W1: identical histograms -> 0",
          wd_core.layer_w1({3: 10, 7: 5}, {3: 20, 7: 10}) < 1e-12)
    check("layer W1: all mass moved 5 layers -> 5",
          abs(wd_core.layer_w1({0: 10}, {5: 10}) - 5) < 1e-12)
    check("layer W1: half the mass moved 4 layers -> 2",
          abs(wd_core.layer_w1({0: 10}, {0: 5, 4: 5}) - 2) < 1e-12)
    uni = wd_core.layer_w1_to_uniform({l: 1 for l in range(32)}, 32)
    check("layer W1 to uniform of a uniform profile == 0", uni < 1e-12)


def test_selections():
    print("\nSelections and the degenerate layer-matched null for layer W1")
    budget = ns.DEFAULT_BUDGET
    available = [m for m in ns.ALL_METHODS
                 if os.path.exists(ns.selection_path(m, budget))]
    check("at least two selections available", len(available) >= 2, str(available))
    if len(available) < 2:
        return available
    rng = np.random.default_rng(0)
    sel = ns.load_selection(available[0], budget)
    rnd = ns.random_layer_matched(sel, rng)
    check("layer W1(selection, its layer-matched null) == 0 -- location is "
          "exactly what that null divides out",
          wd_core.layer_w1(ns.layer_counts(sel), ns.layer_counts(rnd)) < 1e-12)
    return [m for m in ns.DEFAULT_METHODS if m in available]


def test_scripts(methods):
    print("\nEnd-to-end run on synthetic activations")
    if len(methods) < 2:
        check("two selections for the end-to-end run", False, str(methods))
        return
    methods = methods[:3]
    n_pairs = len(methods) * (len(methods) - 1) // 2
    tmp = tempfile.mkdtemp(prefix="wd_smoke_")
    try:
        acts_path = cka_make_synthetic(os.path.join(tmp, "activations"))
        results = os.path.join(tmp, "results")
        env = dict(os.environ, MPLBACKEND="Agg")
        budget = str(ns.DEFAULT_BUDGET)

        cmd = [sys.executable, os.path.join(HERE, "run_wasserstein.py"),
               "--layer_only", "--methods", *methods, "--budget", budget,
               "--output_dir", results]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("run_wasserstein.py --layer_only exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        check("layer W1 CSV written",
              os.path.exists(os.path.join(results, f"wd_layer_selections_N{budget}.csv")))

        cmd = [sys.executable, os.path.join(HERE, "run_wasserstein.py"),
               "--activations", acts_path, "--budget", budget,
               "--methods", *methods, "--null_seeds", "3", "--ceiling_seeds", "2",
               "--variants", "raw", "class+length", "--save_matching",
               "--output_dir", results]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("run_wasserstein.py exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return

        csv = os.path.join(results, f"wd_synthetic_mean_N{budget}.csv")
        check("pairwise CSV written", os.path.exists(csv))
        if not os.path.exists(csv):
            return
        frame = pd.read_csv(csv)
        check("all pairs x variants present", len(frame) == 2 * n_pairs,
              f"rows={len(frame)}")
        check("wasserstein in [0, 1] and matched_corr == 1 - W",
              bool(frame["wasserstein"].between(0, 1).all())
              and bool(((frame["wasserstein"] + frame["matched_corr"]) - 1).abs().max() < 1e-9))
        check("nulls non-trivial (shared-factor data)",
              bool((frame["wd_null_layer_matched_mean"] > 0.05).all()),
              f"min null={frame['wd_null_layer_matched_mean'].min():.3f}")
        # Synthetic neurons are all exchangeable, so a method's own halves are
        # no more matchable than random halves: the excess must be ~0 (and
        # wd_normalized therefore NaN -- the guard in normalized_score).
        check("ceiling ~ half-size null when selections are exchangeable with "
              "random neurons",
              bool((frame["wd_ceiling_excess"].abs() < 0.05).all()),
              f"max |excess|={frame['wd_ceiling_excess'].abs().max():.3f}")
        check("CKA columns present and in [0, 1]",
              bool(frame["cka"].between(-0.01, 1.01).all()))
        check("frac_identical matches Jaccard direction",
              bool(((frame["jaccard"] > 0) == (frame["frac_identical"] > 0)).all()))
        for f in (f"wd_matrix_synthetic_mean_N{budget}_raw.png",
                  f"wd_pairs_synthetic_mean_N{budget}_raw.png",
                  f"wd_vs_cka_synthetic_mean_N{budget}_class-length.png",
                  f"wd_variants_synthetic_mean_N{budget}.png",
                  f"wd_synthetic_mean_N{budget}.json"):
            check(f"{f} written", os.path.exists(os.path.join(results, f)))
        matchings = os.listdir(os.path.join(results, "matchings"))
        check("matching files written (one per pair x variant)",
              len(matchings) == 2 * n_pairs, f"{len(matchings)}")

        cmd = [sys.executable, os.path.join(HERE, "compare_to_cka.py"),
               "--wasserstein", csv]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("compare_to_cka.py (internal CKA) exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        check("agreement summary written",
              os.path.exists(os.path.join(results, f"wd_vs_cka_synthetic_mean_N{budget}.csv")))

        # External-CKA path: fabricate a run_cka.py-shaped table from the
        # internal columns and make sure the merge and the sanity check work.
        fake = frame[["variant", "method_a", "method_b", "cka",
                      "cka_z_vs_null", "cka_normalized"]].rename(
            columns={"cka_z_vs_null": "z_vs_null", "cka_normalized": "normalized"})
        fake_path = os.path.join(results, "cka_fake.csv")
        fake.to_csv(fake_path, index=False)
        proc = subprocess.run(cmd + ["--cka", fake_path], capture_output=True,
                              text=True, env=env)
        check("compare_to_cka.py (external CKA table) exits cleanly",
              proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        check("external path reports matching observed CKA",
              "max |diff| = 0.00e+00" in proc.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("Wasserstein pipeline smoke test")
    test_measures()
    available = test_selections()
    test_scripts(available)
    print(f"\n{'FAILURES: %d' % check.failures if check.failures else 'All checks passed.'}")
    sys.exit(1 if check.failures else 0)

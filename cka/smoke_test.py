"""End-to-end check of the CKA pipeline with synthetic activations -- no GPU,
no model download, runs in seconds.

It also validates the measures against cases with a known answer, which is the
part worth keeping: it is easy to write a CKA function that silently returns
plausible-looking numbers.

    python cka/smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cka_core as core
import neuron_sets as ns

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        check.failures += 1


check.failures = 0


def test_measures():
    print("\nMeasure sanity checks")
    rng = np.random.default_rng(0)
    n = 300

    X = core.prepare(rng.normal(size=(n, 40)), zscore=True)
    check("CKA(X, X) == 1", abs(core.linear_cka(X, X) - 1.0) < 1e-6,
          f"got {core.linear_cka(X, X):.6f}")

    # Invariance to orthogonal transform: the defining property of CKA and the
    # reason it can compare populations whose dimensions do not correspond.
    Q, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    XQ = core.prepare(X @ Q, zscore=False)
    check("CKA invariant to rotation", abs(core.linear_cka(X, XQ) - 1.0) < 1e-5,
          f"got {core.linear_cka(X, XQ):.6f}")

    # Invariance to isotropic scaling (but NOT to per-column scaling, which is
    # exactly why the z-score step exists).
    check("CKA invariant to isotropic scale",
          abs(core.linear_cka(X, core.prepare(X * 7.3, zscore=False)) - 1.0) < 1e-6)

    Y = core.prepare(rng.normal(size=(n, 55)), zscore=True)
    indep = core.linear_cka(X, Y)
    check("CKA of independent matrices is small", indep < 0.35, f"got {indep:.4f}")

    unb = core.linear_cka_unbiased(X, Y)
    check("unbiased CKA of independent matrices is smaller than biased",
          abs(unb) < indep, f"biased={indep:.4f} unbiased={unb:.4f}")

    # Shared latent structure must be detected across different widths.
    latent = rng.normal(size=(n, 5))
    A = core.prepare(latent @ rng.normal(size=(5, 30)) + 0.1 * rng.normal(size=(n, 30)))
    B = core.prepare(latent @ rng.normal(size=(5, 70)) + 0.1 * rng.normal(size=(n, 70)))
    shared = core.linear_cka(A, B)
    check("CKA detects shared latent across different widths", shared > 0.8,
          f"got {shared:.4f}")
    rho = core.rsa_spearman(A, B)
    check("RSA agrees on the shared-latent case", rho > 0.6, f"got {rho:.4f}")

    # Residualization must actually annihilate the nuisance direction.
    labels = np.repeat([0, 1], n // 2)
    signal = np.outer(labels.astype(float), rng.normal(size=25)) * 6.0
    noise = rng.normal(size=(n, 25))
    Z = core.design_matrix(labels=labels)
    res = core.prepare(signal + noise, Z=Z, zscore=False)
    class_gap = abs(res[labels == 0].mean() - res[labels == 1].mean())
    check("class residualization removes the class means", class_gap < 1e-4,
          f"gap={class_gap:.2e}")

    lengths = rng.integers(20, 200, size=n)
    Z2 = core.design_matrix(labels=labels, token_counts=lengths)
    lenX = signal + np.outer(lengths.astype(float), rng.normal(size=25)) + noise
    res2 = core.prepare(lenX, Z=Z2, zscore=False)
    corr = max(abs(np.corrcoef(lengths, res2[:, j])[0, 1]) for j in range(25))
    check("length residualization removes the length axis", corr < 1e-6,
          f"max |corr| = {corr:.2e}")

    # Near-constant (dead) neurons must be dropped, not amplified by z-scoring.
    with_dead = np.concatenate([rng.normal(size=(n, 10)), np.zeros((n, 3))], axis=1)
    kept, dropped = core.prepare(with_dead, zscore=True, return_dropped=True)
    check("dead neurons dropped", dropped == 3 and kept.shape[1] == 10,
          f"dropped={dropped} shape={kept.shape}")

    # The two implementations of ||X^T Y||_F^2 must agree, since the code picks
    # between them based on matrix sizes.
    big = core.prepare(rng.normal(size=(60, 200)))
    small = core.prepare(rng.normal(size=(60, 5)))
    feature_form = float(np.sum((big.T @ small).astype(np.float64) ** 2))
    gram_form = float(np.sum((big @ big.T).astype(np.float64)
                             * (small @ small.T).astype(np.float64)))
    check("feature-form and Gram-form cross norms agree",
          abs(feature_form - gram_form) / max(feature_form, 1e-9) < 1e-5)


def test_selections():
    print("\nNeuron selection loaders")
    for budget in ns.BUDGETS:
        for method in ns.ALL_METHODS:
            try:
                sel = ns.load_selection(method, budget)
            except FileNotFoundError:
                check(f"{method} @ N={budget}", False, "file missing")
                continue
            total = ns.size(sel)
            ok = abs(total - budget) <= 3  # zhao_topk files are 1-2 rows short
            check(f"{method} @ N={budget}", ok, f"total={total}")

    rng = np.random.default_rng(0)
    sel = ns.load_selection("siren", 2500)
    rnd = ns.random_layer_matched(sel, rng)
    check("layer-matched null preserves per-layer counts",
          ns.layer_counts(sel) == ns.layer_counts(rnd))
    glob = ns.random_global(sel, rng)
    check("global null preserves the total size", ns.size(glob) == ns.size(sel))
    h1, h2 = ns.split_halves(sel, rng)
    overlap = ns.jaccard(h1, h2)
    check("disjoint halves really are disjoint", overlap == 0.0,
          f"jaccard={overlap}")
    check("halves are roughly half the size",
          abs(ns.size(h1) - ns.size(sel) / 2) < 40,
          f"{ns.size(h1)} vs {ns.size(sel) / 2:.0f}")


def make_synthetic(out_dir, n_prompts=260, seed=0):
    """Activations with a KNOWN answer: a shared global factor plus a
    class signal plus a length signal, so every neuron in the model shares
    structure. All methods should land near their ceiling, and the layer-matched
    null should be high too -- which is precisely the point the controls make.
    """
    rng = np.random.default_rng(seed)
    n_layers, width = ns.NUM_LAYERS, ns.INTERMEDIATE_SIZE
    labels = np.repeat([0, 1], n_prompts // 2)
    lengths = rng.integers(20, 200, size=n_prompts)

    latent = rng.normal(size=(n_prompts, 4))
    acts = np.zeros((n_prompts, n_layers, width), dtype=np.float16)
    for layer in range(n_layers):
        loadings = rng.normal(size=(4, width)) * 0.8
        block = (latent @ loadings
                 + np.outer(labels.astype(np.float32), rng.normal(size=width))
                 + np.outer(lengths.astype(np.float32) / 100.0,
                            rng.normal(size=width))
                 + rng.normal(size=(n_prompts, width)) * 0.5)
        acts[:, layer] = block.astype(np.float16)

    os.makedirs(out_dir, exist_ok=True)
    acts_path = os.path.join(out_dir, "synthetic_mean.npy")
    np.save(acts_path, acts)
    pd.DataFrame({
        "label": labels, "n_tokens": lengths,
        "dataset": "synthetic", "text": [f"prompt {i}" for i in range(n_prompts)],
    }).to_csv(os.path.join(out_dir, "synthetic_mean.meta.csv"), index=False)
    return acts_path


def test_scripts():
    print("\nEnd-to-end run on synthetic activations")
    tmp = tempfile.mkdtemp(prefix="cka_smoke_")
    try:
        acts_path = make_synthetic(os.path.join(tmp, "activations"))
        results = os.path.join(tmp, "results")
        env = dict(os.environ, MPLBACKEND="Agg")

        cmd = [sys.executable, os.path.join(HERE, "run_cka.py"),
               "--activations", acts_path, "--budget", "2500",
               "--null_seeds", "3", "--ceiling_seeds", "2",
               "--variants", "raw", "class+length",
               "--skip_rsa", "--output_dir", results]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("run_cka.py exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return

        csv = os.path.join(results, "cka_synthetic_mean_N2500.csv")
        check("pairwise CSV written", os.path.exists(csv))
        if os.path.exists(csv):
            frame = pd.read_csv(csv)
            check("all method pairs present, both variants",
                  len(frame) == 12, f"rows={len(frame)}")
            check("CKA values in [0, 1]",
                  bool(frame["cka"].between(-0.01, 1.01).all()))
            check("nulls are non-trivial on shared-factor data (this is why "
                  "controls matter)",
                  bool((frame["null_layer_matched_mean"] > 0.1).all()),
                  f"min null={frame['null_layer_matched_mean'].min():.3f}")
            check("ceiling below 1.0",
                  bool((frame["ceiling_mean"] < 1.0).all()))
            raw = frame[frame["variant"] == "raw"]["cka"].mean()
            res = frame[frame["variant"] == "class+length"]["cka"].mean()
            check("residualization changes the answer", abs(raw - res) > 1e-4,
                  f"raw={raw:.4f} class+length={res:.4f}")

        cmd = [sys.executable, os.path.join(HERE, "run_cross_layer.py"),
               "--activations", acts_path, "--budget", "2500",
               "--methods", "siren", "wang", "--variant", "class+length",
               "--null_seeds", "2", "--output_dir", results]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("run_cross_layer.py exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return
        stem = "crosslayer_synthetic_mean_N2500_class-length_siren_vs_wang"
        check("cross-layer CKA CSV written",
              os.path.exists(os.path.join(results, stem + "_cka.csv")))
        check("cross-layer plot written",
              os.path.exists(os.path.join(results, stem + ".png")))
        cl = pd.read_csv(os.path.join(results, stem + "_cka.csv"), index_col=0)
        check("cross-layer matrix is 32x32", cl.shape == (32, 32),
              f"shape={cl.shape}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("CKA pipeline smoke test")
    test_measures()
    test_selections()
    test_scripts()
    print(f"\n{'FAILURES: %d' % check.failures if check.failures else 'All checks passed.'}")
    sys.exit(1 if check.failures else 0)

"""End-to-end check of the refusal-direction / DFA pipeline with synthetic
data -- no GPU, no model download, runs in under a minute.

Validates the math against cases with known answers (planted directions,
dense recomputation of the streamed accumulations), then runs
fit_direction.py and run_dfa.py as subprocesses on synthetic residuals,
synthetic activations and a synthetic .npz weight file, using the REAL
neuron-selection files.

    python refusal_direction/smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cka"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import downproj
import neuron_sets as ns
import refusal_core as core

HERE = os.path.dirname(os.path.abspath(__file__))

D_MODEL = 8          # tiny residual stream; width must stay real (14336)
NUM_HIDDEN = ns.NUM_LAYERS + 1


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        check.failures += 1


check.failures = 0


def test_core():
    print("\nCore math")
    rng = np.random.default_rng(0)

    # AUROC on known cases.
    check("AUROC of perfect separation is 1",
          core.auroc([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1]) == 1.0)
    check("AUROC of inverted separation is 0",
          core.auroc([10, 11, 12, 1, 2, 3], [0, 0, 0, 1, 1, 1]) == 0.0)
    a = core.auroc(rng.normal(size=2000), rng.integers(0, 2, size=2000))
    check("AUROC of noise is ~0.5", abs(a - 0.5) < 0.05, f"got {a:.3f}")
    check("AUROC handles ties",
          core.auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5)

    # diff_in_means recovers a planted direction.
    n = 400
    labels = np.repeat([0, 1], n // 2)
    planted = core.unit(rng.normal(size=D_MODEL))
    resid = (rng.normal(size=(n, NUM_HIDDEN, D_MODEL)) * 0.3
             + labels[:, None, None] * planted[None, None, :] * 2.0)
    fitted = core.diff_in_means(resid, labels)
    cos = core.cosine(fitted[10], planted)
    check("diff-in-means recovers the planted direction", cos > 0.99,
          f"cos={cos:.4f}")

    # DFA identity: summing per-neuron scores equals projecting the summed
    # MLP write. This is the linearity the whole decomposition rests on.
    W = rng.normal(size=(D_MODEL, 100))
    delta_a = rng.normal(size=100)
    r_hat = core.unit(rng.normal(size=D_MODEL))
    per_neuron = float(np.sum(delta_a * (r_hat @ W)))
    aggregated = float((W @ delta_a) @ r_hat)
    check("per-neuron DFA sums to the projected write vector",
          abs(per_neuron - aggregated) < 1e-9,
          f"|diff|={abs(per_neuron - aggregated):.2e}")

    # Layer selection respects the depth cap.
    aurocs = np.linspace(0.5, 0.99, NUM_HIDDEN)  # best is the LAST state
    chosen = core.choose_hidden(aurocs, max_frac=0.8)
    check("depth cap excludes late hidden states",
          chosen == int(np.ceil(0.8 * NUM_HIDDEN)) - 1, f"chosen={chosen}")

    # Refusal detection.
    check("refusal detected",
          core.is_refusal("I'm sorry, but I cannot help with that request."))
    check("compliance not flagged",
          not core.is_refusal("Sure! Here is a step-by-step recipe."))

    # Direction round trip.
    tmp = tempfile.mkdtemp(prefix="refdir_smoke_")
    try:
        path = os.path.join(tmp, "d.npz")
        core.save_direction(path, fitted, aurocs, aurocs, 10, "model-x",
                            "tag-y", ns.NUM_LAYERS)
        loaded = core.load_direction(path)
        check("direction round trip",
              loaded["chosen"] == 10
              and np.allclose(loaded["directions"], fitted)
              and abs(np.linalg.norm(loaded["unit"][10]) - 1.0) < 1e-5
              and loaded["model_path"] == "model-x")
        check("direction_for_layer modes",
              np.allclose(core.direction_for_layer(loaded, 5, "single"),
                          loaded["unit"][10])
              and np.allclose(core.direction_for_layer(loaded, 5, "per_layer"),
                              loaded["unit"][6]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_downproj_npz():
    print("\nWeight streaming (npz path)")
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp(prefix="refdir_smoke_")
    try:
        arrays = {f"layer{l}": rng.normal(size=(D_MODEL, 20)).astype(np.float32)
                  for l in range(4)}
        path = os.path.join(tmp, "w.npz")
        np.savez(path, **arrays)
        seen = dict(downproj.iter_down_proj(path))
        check("all layers yielded once", sorted(seen) == [0, 1, 2, 3])
        check("weights round trip",
              all(np.allclose(seen[l], arrays[f"layer{l}"]) for l in seen))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_synthetic(tmp, rng, n_prompts=200):
    """Residuals with the separation peaking at h=20 (and a decoy beyond the
    depth cap at h=30), plus activations and weights for the DFA step."""
    labels = np.repeat([0, 1], n_prompts // 2)

    strength = np.zeros(NUM_HIDDEN)
    for h in range(NUM_HIDDEN):
        strength[h] = 2.5 * np.exp(-((h - 20) ** 2) / 8.0)
    strength[30] = 4.0  # stronger, but beyond the 80% depth cap
    planted = core.unit(rng.normal(size=(NUM_HIDDEN, D_MODEL)), axis=1)
    def residuals(n_rows, row_labels):
        base = rng.normal(size=(n_rows, NUM_HIDDEN, D_MODEL)) * 0.6
        return (base + row_labels[:, None, None] * strength[None, :, None]
                * planted[None, :, :]).astype(np.float16)

    resid_dir = os.path.join(tmp, "residuals")
    os.makedirs(resid_dir)
    paths = {}
    for split, n_rows in (("fit", n_prompts), ("val", n_prompts // 2)):
        row_labels = np.repeat([0, 1], n_rows // 2)
        path = os.path.join(resid_dir, f"synth_{split}_last.npy")
        np.save(path, residuals(n_rows, row_labels))
        pd.DataFrame({
            "label": row_labels, "n_tokens": 30, "dataset": "synth",
            "text": [f"{split} {i}" for i in range(n_rows)],
        }).to_csv(os.path.join(resid_dir, f"synth_{split}_last.meta.csv"),
                  index=False)
        paths[split] = path

    acts_dir = os.path.join(tmp, "activations")
    os.makedirs(acts_dir)
    acts = (rng.normal(size=(n_prompts, ns.NUM_LAYERS, ns.INTERMEDIATE_SIZE))
            * 0.5
            + labels[:, None, None]
            * rng.normal(size=(ns.NUM_LAYERS, ns.INTERMEDIATE_SIZE))[None]
            * 0.5).astype(np.float16)
    acts_path = os.path.join(acts_dir, "synth_last.npy")
    np.save(acts_path, acts)
    pd.DataFrame({
        "label": labels, "n_tokens": 30, "dataset": "synth",
        "text": [f"prompt {i}" for i in range(n_prompts)],
    }).to_csv(os.path.join(acts_dir, "synth_last.meta.csv"), index=False)

    weights_path = os.path.join(tmp, "weights.npz")
    np.savez(weights_path,
             **{f"layer{l}": (rng.normal(size=(D_MODEL, ns.INTERMEDIATE_SIZE))
                              * 0.05).astype(np.float32)
                for l in range(ns.NUM_LAYERS)})
    return paths, acts_path, acts, labels, weights_path


def test_scripts():
    print("\nEnd-to-end run on synthetic data")
    methods = [m for m in ns.DEFAULT_METHODS
               if os.path.exists(ns.selection_path(m, ns.DEFAULT_BUDGET))]
    if len(methods) < 2:
        check("at least two selections available", False, f"found {methods}")
        return
    rng = np.random.default_rng(2)
    tmp = tempfile.mkdtemp(prefix="refdir_smoke_")
    try:
        resid_paths, acts_path, acts, labels, weights_path = \
            make_synthetic(tmp, rng)
        results = os.path.join(tmp, "results")
        directions = os.path.join(tmp, "directions")
        env = dict(os.environ, MPLBACKEND="Agg")

        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "fit_direction.py"),
             "--fit_residuals", resid_paths["fit"],
             "--val_residuals", resid_paths["val"],
             "--model_path", weights_path,
             "--directions_dir", directions, "--output_dir", results],
            capture_output=True, text=True, env=env)
        check("fit_direction.py exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
            return

        dir_path = os.path.join(directions, "synth_last.npz")
        check("direction artifact written", os.path.exists(dir_path))
        dir_data = core.load_direction(dir_path)
        check("chosen layer is the planted one (and not the decoy beyond "
              "the cap)", abs(dir_data["chosen"] - 20) <= 1,
              f"chosen={dir_data['chosen']}")

        budget = str(ns.DEFAULT_BUDGET)
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "run_dfa.py"),
             "--activations", acts_path, "--direction", dir_path,
             "--weights", weights_path, "--methods", *methods,
             "--budget", budget, "--null_seeds", "3",
             "--scalar_null_seeds", "10", "--output_dir", results],
            capture_output=True, text=True, env=env)
        check("run_dfa.py exits cleanly", proc.returncode == 0,
              proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
        if proc.returncode:
            print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
            return

        tag = f"synth_last_dir-synth_last_N{budget}"
        for stem in (f"dfa_summary_{tag}.csv", f"writevec_cos_{tag}.csv",
                     f"dfa_neurons_{tag}.csv", f"writevecs_{tag}.npz",
                     f"dfa_map_synth_last_dir-synth_last.npy",
                     f"writevec_matrix_{tag}.png",
                     f"direction_alignment_{tag}.png"):
            check(f"output {stem}", os.path.exists(os.path.join(results, stem)))

        # Dense recomputation: the streamed DFA map and write vectors must
        # equal what a direct (all-in-memory) computation gives.
        harm = labels == 1
        delta = np.stack([
            acts[harm, l, :].astype(np.float64).mean(axis=0)
            - acts[~harm, l, :].astype(np.float64).mean(axis=0)
            for l in range(ns.NUM_LAYERS)]).astype(np.float32)
        weights = dict(downproj.iter_down_proj(weights_path))
        r_hat = dir_data["unit"][dir_data["chosen"]]

        dfa_map = np.load(os.path.join(
            results, "dfa_map_synth_last_dir-synth_last.npy"))
        manual_map = np.stack([delta[l] * (r_hat @ weights[l])
                               for l in range(ns.NUM_LAYERS)])
        err = np.max(np.abs(dfa_map - manual_map))
        check("streamed DFA map matches dense recomputation", err < 1e-4,
              f"max |diff| = {err:.2e}")

        vecs = np.load(os.path.join(results, f"writevecs_{tag}.npz"))
        m = methods[0]
        sel = ns.load_selection(m, ns.DEFAULT_BUDGET)
        manual_v = np.zeros(D_MODEL)
        for layer, idx in sel.items():
            manual_v += weights[layer][:, idx].astype(np.float64) @ delta[layer, idx]
        rel = (np.linalg.norm(vecs[m] - manual_v)
               / max(np.linalg.norm(manual_v), 1e-12))
        check(f"streamed write vector matches dense recomputation ({m})",
              rel < 1e-5, f"rel err = {rel:.2e}")

        summary = pd.read_csv(os.path.join(results, f"dfa_summary_{tag}.csv"))
        check("summary has one row per method", len(summary) == len(methods))
        check("summary values finite",
              bool(np.isfinite(summary[["dfa_sum", "cos_r"]].to_numpy()).all()))
        pairs = pd.read_csv(os.path.join(results, f"writevec_cos_{tag}.csv"))
        n_entities = len(methods) + 1  # + all_mlp
        check("all entity pairs present",
              len(pairs) == n_entities * (n_entities - 1) // 2,
              f"rows={len(pairs)}")
        check("cosines in [-1, 1]",
              bool(pairs["cos"].between(-1.0001, 1.0001).all()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("Refusal-direction / DFA pipeline smoke test")
    test_core()
    test_downproj_npz()
    test_scripts()
    print(f"\n{'FAILURES: %d' % check.failures if check.failures else 'All checks passed.'}")
    sys.exit(1 if check.failures else 0)

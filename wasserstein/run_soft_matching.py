"""Soft Matching on AdvBench, Khosla & Williams (arXiv:2311.09466).

Isolated from run_wasserstein.py so WildGuard / older W+CKA runs stay intact.

For each method, every selected neuron is its raw activation profile on the
same AdvBench prompts. Pairwise comparison is a square assignment under

    cost(i, j) = 1 - corr(A_i, B_j)     # SIGNED Pearson, not |corr|

S(A, B) is the mean matched correlation. One control: 20 layer-matched
random draws (same per-layer counts as each method). Raw activations only.

Writes exactly two files:
    wasserstein/results/soft_matching_advbench_N4588.png
    wasserstein/results/soft_matching_advbench_N4588.csv

    python wasserstein/run_soft_matching.py \
        --activations cka/activations/advbench_mean.npy
"""
import argparse
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "cka"))
sys.path.append(HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neuron_sets as ns
from run_cka import load_activations

import wd_core

DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")
BUDGET = 4588
METHODS = tuple(ns.ALL_METHODS)
NULL_SEEDS = 20


def trim_to_n(sel, n):
    """Drop extras so the selection has exactly n neurons (Zhao ties)."""
    total = ns.size(sel)
    if total == n:
        return sel
    if total < n:
        raise SystemExit(f"selection has {total} neurons, need {n}")
    layers, idx = ns.flatten(sel)
    keep_l, keep_i = layers[:n], idx[:n]
    out = {}
    for layer, neuron in zip(keep_l.tolist(), keep_i.tolist()):
        out.setdefault(int(layer), []).append(int(neuron))
    return {layer: np.asarray(v, dtype=np.int64) for layer, v in out.items()}


def replace_dead(acts, sel, rng, width, name):
    """Swap near-constant neurons for unused ones in the same layer.

    Keeps per-layer counts (and therefore N) fixed so every method stays at
    the same square size. Reports how many replacements were needed.
    """
    replaced = 0
    for _ in range(32):
        X = ns.build_matrix(acts, sel)
        keep = wd_core.keep_mask(X)
        if keep.all():
            return sel, X, replaced
        layers, idx = ns.flatten(sel)
        used = {layer: set(sel[layer].tolist()) for layer in sel}
        new = {layer: np.array(sel[layer], copy=True) for layer in sel}
        for col in np.where(~keep)[0]:
            layer = int(layers[col])
            old = int(idx[col])
            pool = np.array([i for i in range(width) if i not in used[layer]],
                            dtype=np.int64)
            if pool.size == 0:
                raise SystemExit(f"{name}: layer {layer} has no unused neurons "
                                 f"to replace a dead unit")
            nxt = int(rng.choice(pool))
            used[layer].discard(old)
            used[layer].add(nxt)
            pos = np.where(new[layer] == old)[0]
            if pos.size != 1:
                raise SystemExit(f"{name}: layer {layer} neuron {old} not unique")
            new[layer][int(pos[0])] = nxt
            replaced += 1
        sel = {layer: np.sort(idx) for layer, idx in new.items()}
    raise SystemExit(f"{name}: still have near-constant neurons after replacements")


def valid_matrix(acts, sel, rng, width, n, name):
    sel = trim_to_n(sel, n)
    sel, X, replaced = replace_dead(acts, sel, rng, width, name)
    if X.shape[1] != n:
        raise SystemExit(f"{name}: expected {n} columns, got {X.shape[1]}")
    if not wd_core.keep_mask(X).all():
        raise SystemExit(f"{name}: dead neurons remain")
    return sel, X, replaced


def two_panel(obs, null, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    panels = (
        (obs, "Soft Matching correlation | AdvBench | Top 1% (N=4588)"),
        (null, "Layer-matched random baseline"),
    )
    for ax, matrix, title in zip(axes, *zip(*panels)):
        finite = matrix[np.isfinite(matrix)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        mid = float(np.nanmean(matrix)) if finite.size else 0.5
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > mid else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", required=True,
                        help="cka/activations/advbench_mean.npy")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--null_seeds", type=int, default=NULL_SEEDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.budget != BUDGET:
        print(f"NOTE: this script is the AdvBench N={BUDGET} experiment; "
              f"running at N={args.budget} as requested.")
    methods = list(args.methods)
    if methods == ["all"]:
        methods = list(METHODS)
    for m in methods:
        if m not in ns.METHOD_SPECS:
            raise SystemExit(f"unknown method '{m}'")
    if len(methods) < 2:
        raise SystemExit("need at least two methods")

    os.makedirs(args.output_dir, exist_ok=True)
    acts, meta = load_activations(args.activations)
    n_prompts, n_layers, width = acts.shape
    print(f"Activations: {n_prompts} prompts x {n_layers} layers x {width}")
    print(f"  labels: {meta['label'].value_counts().to_dict()}")
    print("  raw activations only; signed Pearson cost 1 - corr")
    print(f"  Soft Matching (Khosla & Williams), N={args.budget}, "
          f"{args.null_seeds} layer-matched seeds")

    rng_fix = np.random.default_rng(args.seed)
    selections, matrices = {}, {}
    for m in methods:
        raw_sel = ns.load_selection(m, args.budget)
        raw_n = ns.size(raw_sel)
        sel, X, replaced = valid_matrix(
            acts, raw_sel, rng_fix, width, args.budget, ns.display_name(m))
        selections[m] = sel
        matrices[m] = X
        extra = f"  trimmed {raw_n}->{args.budget}" if raw_n != args.budget else ""
        dead = f"  replaced {replaced} dead" if replaced else ""
        print(f"  {ns.display_name(m):18s} {X.shape[1]} neurons"
              f"{extra}{dead}")

    sizes = {m: matrices[m].shape[1] for m in methods}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"unequal usable sizes, refusing rectangular OT: {sizes}")
    n_neurons = next(iter(sizes.values()))
    print(f"  all methods have exactly {n_neurons} usable neurons")

    # Observed: S(X, X) is 1 by construction; do not re-assign the diagonal.
    k = len(methods)
    m_idx = {m: i for i, m in enumerate(methods)}
    obs = np.full((k, k), np.nan)
    for i, m in enumerate(methods):
        obs[i, i] = 1.0
        self_s = wd_core.soft_matching(matrices[m], matrices[m])
        if abs(self_s - 1.0) > 1e-5:
            raise SystemExit(f"{m}: S(X,X)={self_s:.6f}, expected 1")

    pairs = list(itertools.combinations(methods, 2))
    observed = {}
    print("\nObserved Soft Matching (signed)")
    for a, b in pairs:
        s = wd_core.soft_matching(matrices[a], matrices[b])
        observed[(a, b)] = s
        i, j = m_idx[a], m_idx[b]
        obs[i, j] = obs[j, i] = s
        print(f"  {ns.display_name(a):16s} / {ns.display_name(b):16s}  {s:.4f}")

    print(f"\nLayer-matched random ({args.null_seeds} seeds)")
    rng_master = np.random.default_rng(args.seed + 1)
    null_vals = {p: [] for p in pairs}
    for s in range(args.null_seeds):
        rng = np.random.default_rng(rng_master.integers(1 << 62))
        rand_X = {}
        for m in methods:
            rnd = ns.random_layer_matched(selections[m], rng, width)
            rand_X[m] = ns.build_matrix(acts, rnd)
            if rand_X[m].shape[1] != n_neurons:
                raise SystemExit(f"null {m}: {rand_X[m].shape[1]} != {n_neurons}")
        for a, b in pairs:
            null_vals[(a, b)].append(wd_core.soft_matching(rand_X[a], rand_X[b]))
        print(f"  draw {s + 1}/{args.null_seeds}", end="\r", flush=True)
    print(" " * 40, end="\r")

    null = np.full((k, k), np.nan)
    rows = []
    print("\nPairwise")
    for a, b in pairs:
        vals = np.asarray(null_vals[(a, b)], dtype=np.float64)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        i, j = m_idx[a], m_idx[b]
        null[i, j] = null[j, i] = mu
        s = observed[(a, b)]
        rows.append({
            "method_a": a, "method_b": b, "n_neurons": n_neurons,
            "soft_matching_corr": s,
            "layer_matched_random_mean": mu,
            "layer_matched_random_std": sd,
        })
        print(f"  {ns.display_name(a):16s} / {ns.display_name(b):16s}  "
              f"S={s:.4f}   layer-matched random {mu:.4f} +/- {sd:.4f}")

    stem = os.path.join(args.output_dir, f"soft_matching_advbench_N{args.budget}")
    csv_path = stem + ".csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    two_panel(obs, null, [ns.display_name(m) for m in methods], stem + ".png")


if __name__ == "__main__":
    main()

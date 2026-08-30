"""Headline analysis: do the four methods' neurons feed the SAME refusal
mechanism, even though they select almost disjoint neurons?

Where cka/run_cka.py asks whether the selected populations induce the same
representational geometry, this asks a mechanistic question: does each
selected neuron differentially WRITE along the refusal direction r_hat
(Arditi et al. 2024) when the prompt is harmful?

Per neuron (l, j), the contribution to the residual stream is
a_{l,j}(x) * W_down^(l)[:, j], so its class-contrast DFA score is

    DFA_{l,j} = (E_harmful[a_{l,j}] - E_benign[a_{l,j}]) * (W_down^(l)[:, j] . r_hat)

computed on held-out prompts (the same activation files the CKA analysis
uses). Per method M, the aggregate residual-stream write vector is

    v_M = sum_{(l,j) in S_M} (E_H[a_{l,j}] - E_B[a_{l,j}]) * W_down^(l)[:, j]

The 4x4 cosine matrix between the v_M -- plus each cos(v_M, r_hat) -- is the
functional analogue of the CKA matrix: two methods can share ZERO neurons and
still have cos(v_A, v_B) near 1 if their neurons write the same direction.

Controls (raw numbers are never reported alone):
  - layer-matched random null: random neurons with each method's per-layer
    counts. The whole MLP writes a large refusal component by itself, so a
    random subset already shows positive alignment; z_vs_null is the
    statistic that matters, exactly as in the CKA analysis.
  - the all-MLP reference vector (every neuron): the ceiling direction that
    any sum of MLP contributions is pulled toward.

Needs no GPU: activations come from cka/extract_activations.py, the
direction from fit_direction.py, and the down_proj weights are streamed from
the local HF cache one layer at a time (see downproj.py).

Usage:
    python refusal_direction/run_dfa.py \
        --activations cka/activations/wildguard_last.npy \
        --direction refusal_direction/directions/openai_moderation_last.npz
"""
import argparse
import itertools
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cka"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import downproj
import neuron_sets as ns
import plots
import refusal_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")


def load_activations(path):
    meta_path = os.path.splitext(path)[0] + ".meta.csv"
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing metadata file {meta_path}")
    acts = np.load(path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    if len(meta) != acts.shape[0]:
        raise SystemExit(f"{path} has {acts.shape[0]} rows but "
                         f"{meta_path} has {len(meta)}")
    return acts, meta


def class_contrast(acts, labels):
    """[num_layers, width] harmful-minus-benign mean activation, layer by
    layer so the fp16 array never has to be fully resident."""
    labels = np.asarray(labels).astype(int)
    if not ((labels == 0).any() and (labels == 1).any()):
        raise SystemExit("the activation set must contain both classes; "
                         "DFA is a harmful-vs-benign contrast")
    n, num_layers, width = acts.shape
    delta = np.zeros((num_layers, width), dtype=np.float32)
    harm = labels == 1
    for layer in range(num_layers):
        block = np.asarray(acts[:, layer, :], dtype=np.float64)
        delta[layer] = (block[harm].mean(axis=0)
                        - block[~harm].mean(axis=0)).astype(np.float32)
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--activations", required=True,
                        help="cka/activations/{tag}_{pooling}.npy. Pooling "
                             "'last' matches the direction's position; 'mean' "
                             "works as a robustness check.")
    parser.add_argument("--direction", required=True,
                        help="refusal_direction/directions/{tag}.npz from "
                             "fit_direction.py")
    parser.add_argument("--weights", default=None,
                        help="Source of the down_proj weights: HF repo id, "
                             "local model dir, or .npz (tests). Defaults to "
                             "the model recorded in the direction artifact.")
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS),
                        help=f"choose from {list(ns.ALL_METHODS)}, or 'all'")
    parser.add_argument("--budget", type=int, default=ns.DEFAULT_BUDGET,
                        choices=ns.BUDGETS)
    parser.add_argument("--direction_mode", default="single",
                        choices=["single", "per_layer"],
                        help="'single' projects every layer's writes onto the "
                             "one chosen direction (the paper's usage); "
                             "'per_layer' uses the direction fitted at each "
                             "layer's own output, guarding against rotation "
                             "across depth.")
    parser.add_argument("--null_seeds", type=int, default=20,
                        help="Layer-matched random draws that get full write "
                             "vectors (streamed with the weights).")
    parser.add_argument("--scalar_null_seeds", type=int, default=200,
                        help="Extra draws for the per-neuron scalar null; "
                             "these only index the DFA map, so they are free.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None,
                        help="Output filename tag; defaults to the activation "
                             "file's basename + the direction tag.")
    args = parser.parse_args()

    methods = list(ns.ALL_METHODS) if args.methods == ["all"] else args.methods
    for m in methods:
        if m not in ns.METHOD_SPECS:
            raise SystemExit(f"unknown method '{m}'; known: {list(ns.ALL_METHODS)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------- inputs
    acts, meta = load_activations(args.activations)
    n_prompts, num_layers, width = acts.shape
    print(f"Activations: {n_prompts} prompts x {num_layers} layers x "
          f"{width} neurons")
    print(f"  classes: {meta['label'].value_counts().to_dict()}")

    dir_data = core.load_direction(args.direction)
    d_model = dir_data["unit"].shape[1]
    chosen = dir_data["chosen"]
    r_hat = dir_data["unit"][chosen]
    print(f"Direction: {args.direction}")
    print(f"  chosen hidden state {chosen} (output of block {chosen - 1}), "
          f"val AUROC {dir_data['auroc_val'][chosen]:.4f}, d_model {d_model}")
    if dir_data["num_layers"] != num_layers:
        raise SystemExit(f"direction was fitted on {dir_data['num_layers']} "
                         f"layers but activations have {num_layers}")

    weights = args.weights or dir_data["model_path"]
    label = args.label or (
        os.path.splitext(os.path.basename(args.activations))[0]
        + f"_dir-{dir_data['fit_tag']}"
        + ("" if args.direction_mode == "single" else "_perlayer"))

    selections = {m: ns.load_selection(m, args.budget) for m in methods}
    for m in methods:
        print("  " + ns.describe(m, args.budget))

    print("\nClass-contrast activations (harmful - benign mean per neuron) ...")
    delta = class_contrast(acts, meta["label"].to_numpy())

    # Null selections are drawn BEFORE streaming so their write vectors can be
    # accumulated in the same single pass over the weights.
    rng_master = np.random.default_rng(args.seed)
    null_sels = [{m: ns.random_layer_matched(selections[m],
                                             np.random.default_rng(
                                                 rng_master.integers(1 << 62)),
                                             width)
                  for m in methods} for _ in range(args.null_seeds)]

    # ------------------------------------------- one pass over the weights
    print(f"\nStreaming down_proj weights from {weights} ...")
    wproj = np.zeros((num_layers, width), dtype=np.float32)
    v_obs = {m: np.zeros(d_model, dtype=np.float64) for m in methods}
    v_null = [{m: np.zeros(d_model, dtype=np.float64) for m in methods}
              for _ in range(args.null_seeds)]
    v_all = np.zeros(d_model, dtype=np.float64)
    seen = set()
    for layer, W in downproj.iter_down_proj(weights):
        if W.shape != (d_model, width):
            raise SystemExit(f"layer {layer}: down_proj is {W.shape}, expected "
                             f"({d_model}, {width}) -- weights, direction and "
                             f"activations disagree")
        r_layer = core.direction_for_layer(dir_data, layer, args.direction_mode)
        wproj[layer] = r_layer @ W
        v_all += W.astype(np.float64) @ delta[layer].astype(np.float64)
        for m in methods:
            idx = selections[m].get(layer)
            if idx is not None:
                v_obs[m] += W[:, idx].astype(np.float64) @ delta[layer, idx]
            for s in range(args.null_seeds):
                nidx = null_sels[s][m].get(layer)
                if nidx is not None:
                    v_null[s][m] += (W[:, nidx].astype(np.float64)
                                     @ delta[layer, nidx])
        seen.add(layer)
        print(f"  layer {layer:2d} done ({len(seen)}/{num_layers})",
              end="\r", flush=True)
    print(" " * 40, end="\r")
    if seen != set(range(num_layers)):
        raise SystemExit(f"weights source yielded layers {sorted(seen)}, "
                         f"expected 0..{num_layers - 1}")

    dfa_map = delta * wproj  # [num_layers, width]
    map_path = os.path.join(args.output_dir, f"dfa_map_{label}.npy")
    np.save(map_path, dfa_map)
    print(f"Saved {map_path}  (full per-neuron DFA map, "
          f"{dfa_map.nbytes / 1e6:.1f} MB)")

    # --------------------------------------------- per-neuron DFA statistics
    print(f"\n{'=' * 72}\nPer-neuron DFA "
          f"(positive = writes toward refusal more on harmful prompts)\n"
          f"{'=' * 72}")
    scalar_rows = []
    neuron_rows = []
    dist_obs, dist_null = {}, {}
    for m in methods:
        values = core.selection_dfa_values(dfa_map, selections[m])
        null_sums = []
        null_pool = []
        for s in range(args.scalar_null_seeds):
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            nsel = ns.random_layer_matched(selections[m], rng, width)
            null_sums.append(core.selection_dfa_sum(dfa_map, nsel))
            if s < 5:  # a few draws are enough for the distribution plot
                null_pool.append(core.selection_dfa_values(dfa_map, nsel))
        null_sums = np.asarray(null_sums)
        obs_sum = float(values.sum())
        z = ((obs_sum - null_sums.mean()) / null_sums.std(ddof=1)
             if null_sums.std(ddof=1) > 0 else float("nan"))
        row = {
            "method": m, "display": ns.display_name(m),
            "n_neurons": int(values.size),
            "dfa_sum": obs_sum,
            "dfa_mean": float(values.mean()),
            "dfa_median": float(np.median(values)),
            "frac_positive": float((values > 0).mean()),
            "null_sum_mean": float(null_sums.mean()),
            "null_sum_std": float(null_sums.std(ddof=1)),
            "z_sum_vs_null": float(z),
        }
        scalar_rows.append(row)
        dist_obs[ns.display_name(m)] = values
        dist_null[ns.display_name(m)] = np.concatenate(null_pool)
        print(f"  {ns.display_name(m):18s} sum={obs_sum:+10.4f}  "
              f"null={null_sums.mean():+8.4f}+/-{null_sums.std(ddof=1):.4f}  "
              f"z={z:+7.1f}  frac_pos={row['frac_positive']:.3f}")

        layers_arr, idx_arr = ns.flatten(selections[m])
        neuron_rows.append(pd.DataFrame({
            "method": m, "layer": layers_arr, "neuron_index": idx_arr,
            "delta_activation": delta[layers_arr, idx_arr],
            "wproj": wproj[layers_arr, idx_arr],
            "dfa": dfa_map[layers_arr, idx_arr],
        }))

    total_dfa = float(dfa_map.sum())
    print(f"  {'all MLP neurons':18s} sum={total_dfa:+10.4f}  "
          f"(reference: what the entire MLP stack writes)")

    # ------------------------------------------------- write-vector cosines
    print(f"\n{'=' * 72}\nWrite-vector alignment "
          f"(the functional analogue of the CKA matrix)\n{'=' * 72}")
    entities = methods + ["all_mlp"]
    vectors = dict(v_obs, all_mlp=v_all)
    disp = {**{m: ns.display_name(m) for m in methods}, "all_mlp": "All MLP"}

    k = len(entities)
    obs_mat = np.full((k, k), np.nan)
    null_mat = np.full((k, k), np.nan)
    z_mat = np.full((k, k), np.nan)
    np.fill_diagonal(obs_mat, 1.0)
    pair_rows = []
    for a, b in itertools.combinations(entities, 2):
        obs = core.cosine(vectors[a], vectors[b])
        nulls = []
        for s in range(args.null_seeds):
            va = v_null[s][a] if a in v_obs else vectors[a]
            vb = v_null[s][b] if b in v_obs else vectors[b]
            nulls.append(core.cosine(va, vb))
        nulls = np.asarray(nulls)
        z = ((obs - nulls.mean()) / nulls.std(ddof=1)
             if nulls.std(ddof=1) > 0 else float("nan"))
        i, j = entities.index(a), entities.index(b)
        obs_mat[i, j] = obs_mat[j, i] = obs
        null_mat[i, j] = null_mat[j, i] = nulls.mean()
        z_mat[i, j] = z_mat[j, i] = z
        pair_rows.append({
            "entity_a": a, "entity_b": b,
            "pair": f"{disp[a]} / {disp[b]}",
            "cos": obs, "null_mean": float(nulls.mean()),
            "null_std": float(nulls.std(ddof=1)), "z_vs_null": float(z),
        })
        print(f"  {disp[a]:14s} vs {disp[b]:14s} cos={obs:+.4f}  "
              f"null={nulls.mean():+.4f}+/-{nulls.std(ddof=1):.4f}  z={z:+7.1f}")

    print("\nAlignment with the refusal direction r_hat itself:")
    for m in methods:
        obs = core.cosine(v_obs[m], r_hat)
        nulls = np.asarray([core.cosine(v_null[s][m], r_hat)
                            for s in range(args.null_seeds)])
        z = ((obs - nulls.mean()) / nulls.std(ddof=1)
             if nulls.std(ddof=1) > 0 else float("nan"))
        for row in scalar_rows:
            if row["method"] == m:
                row.update(cos_r=obs, cos_r_null_mean=float(nulls.mean()),
                           cos_r_null_std=float(nulls.std(ddof=1)),
                           cos_r_z_vs_null=float(z))
        print(f"  {ns.display_name(m):18s} cos(v, r_hat)={obs:+.4f}  "
              f"null={nulls.mean():+.4f}+/-{nulls.std(ddof=1):.4f}  z={z:+7.1f}")
    cos_all_r = core.cosine(v_all, r_hat)
    print(f"  {'all MLP neurons':18s} cos(v, r_hat)={cos_all_r:+.4f}  "
          f"(ceiling reference)")

    # -------------------------------------------------------------- outputs
    tag = f"{label}_N{args.budget}"
    summary_csv = os.path.join(args.output_dir, f"dfa_summary_{tag}.csv")
    pd.DataFrame(scalar_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")

    pairs_csv = os.path.join(args.output_dir, f"writevec_cos_{tag}.csv")
    pd.DataFrame(pair_rows).to_csv(pairs_csv, index=False)
    print(f"Saved {pairs_csv}")

    neurons_csv = os.path.join(args.output_dir, f"dfa_neurons_{tag}.csv")
    pd.concat(neuron_rows, ignore_index=True).to_csv(neurons_csv, index=False)
    print(f"Saved {neurons_csv}")

    vec_path = os.path.join(args.output_dir, f"writevecs_{tag}.npz")
    np.savez(vec_path, r_hat=r_hat, all_mlp=v_all,
             **{m: v_obs[m] for m in methods})
    print(f"Saved {vec_path}")

    json_path = os.path.join(args.output_dir, f"dfa_{tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "activations": os.path.abspath(args.activations),
            "direction": os.path.abspath(args.direction),
            "direction_mode": args.direction_mode,
            "weights": str(weights), "budget": args.budget,
            "methods": methods, "null_seeds": args.null_seeds,
            "total_mlp_dfa": total_dfa, "cos_all_mlp_r_hat": cos_all_r,
            "summary": scalar_rows, "pairs": pair_rows,
        }, f, indent=2)
    print(f"Saved {json_path}")

    # ---------------------------------------------------------------- plots
    names = [disp[e] for e in entities]
    plots.matrix_panel(
        [obs_mat, null_mat, z_mat], names,
        ["Observed cosine of write vectors",
         "Layer-matched random neurons\n(the floor)",
         "z vs null\n(|z| > 3 is significant)"],
        ["RdBu_r", "RdBu_r", "RdBu_r"],
        os.path.join(args.output_dir, f"writevec_matrix_{tag}.png"),
        suptitle=(f"{label} | N={args.budget} | do the methods write the same "
                  f"refusal signal?"),
        vranges=[(-1, 1), (-1, 1), (None, None)])

    plots.alignment_bars(
        [ns.display_name(m) for m in methods],
        [r["cos_r"] for r in scalar_rows],
        [r["cos_r_null_mean"] for r in scalar_rows],
        [r["cos_r_null_std"] for r in scalar_rows],
        os.path.join(args.output_dir, f"direction_alignment_{tag}.png"),
        f"Alignment of each method's write vector with r_hat | {label}")

    plots.dfa_distributions(
        dist_obs, dist_null,
        os.path.join(args.output_dir, f"dfa_dist_{tag}.png"),
        f"Per-neuron DFA | {label} | N={args.budget}")

    plots.dfa_layer_profile(
        {ns.display_name(m): np.array(
            [dfa_map[l, selections[m][l]].sum() if l in selections[m] else 0.0
             for l in range(num_layers)]) for m in methods},
        os.path.join(args.output_dir, f"dfa_layers_{tag}.png"),
        f"Where each method's refusal-writing happens | {label}")

    # ------------------------------------------------------------ reading aid
    print("\n" + "=" * 72)
    print("How to read this")
    print("=" * 72)
    print("z_sum_vs_null >> 0 : the method's neurons write toward refusal far "
          "more than\n                     random neurons from the same layers "
          "-- they sit on the\n                     refusal circuit.")
    print("cos(v_A, v_B) high AND z >> 0 for a cross-method pair: the two "
          "methods found\n                     different neurons feeding the "
          "SAME functional direction.\n                     This is the "
          "convergence result.")
    print("cos(v_M, r_hat) at the all-MLP level: the method captures the bulk "
          "of the\n                     MLP stack's refusal writing with a "
          "tiny fraction of neurons.")
    print("NOTE the null is usually NOT zero -- the whole MLP writes refusal "
          "on harmful\nprompts, so random subsets inherit some alignment. Only "
          "the excess (z) counts.")
    print("\nCausal follow-up: refusal_direction/run_ablation.py checks whether "
          "removing\nthese neurons actually reduces x . r_hat and refusal "
          "behavior.")


if __name__ == "__main__":
    main()

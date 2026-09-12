"""Exact-rank consensus and quota/round-robin over eligible safety neurons.

Eligibility at budget N is the union of the existing top-N safety-neuron
files. Full rankings are used only to score / order those eligible neurons.
The complete neuron universe is never the selection pool for the ensemble.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from load_rankings import (
    candidate_pool,
    methods_selecting,
    ordered_topn,
    percentile,
    write_selection_csv,
)


def consensus_table(rankings, pairs, method_ids, topn_sets=None):
    """Build a per-neuron table of ranks, percentiles, and mean percentile.

    consensus_score(j) = mean_m (r_m(j) - 1) / (M_m - 1)
    Lower is better.
    """
    n_ranked = {mid: len(df) for mid, df in rankings.items()}
    maps = {}
    for mid, df in rankings.items():
        maps[mid] = {
            (int(l), int(i)): int(r)
            for l, i, r in zip(df["layer"], df["neuron_index"], df["rank"])
        }

    rows = []
    missing = Counter()
    for layer, idx in pairs:
        pair = (int(layer), int(idx))
        row = {"layer": pair[0], "neuron_index": pair[1]}
        percentiles = []
        for mid in method_ids:
            r = maps[mid].get(pair)
            if r is None:
                missing[mid] += 1
                r = n_ranked[mid] + 1
            row[f"rank_{mid}"] = r
            p = percentile(r, n_ranked[mid])
            row[f"percentile_{mid}"] = p
            percentiles.append(p)
        row["consensus_score"] = float(np.mean(percentiles))
        if topn_sets is not None:
            selected = methods_selecting(topn_sets, pair)
            row["n_methods_at_N"] = len(selected)
            row["methods_that_selected_it_at_N"] = ";".join(selected)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["consensus_score", "layer", "neuron_index"], kind="mergesort"
    ).reset_index(drop=True)
    return df, missing


def rank_consensus_at_n(rankings, topn_sets, method_ids, n):
    """Select N eligible neurons with the lowest mean percentile rank."""
    pool = candidate_pool(topn_sets)
    if len(pool) < n:
        raise ValueError(
            f"|C_N|={len(pool)} < N={n}; cannot select {n} eligible neurons"
        )
    table, missing = consensus_table(rankings, sorted(pool), method_ids, topn_sets)
    selected = table.head(n).copy()
    selected.insert(0, "variant", "rank_consensus")
    selected.insert(1, "selection_rank", np.arange(1, n + 1))
    extras = (
        [f"rank_{m}" for m in method_ids]
        + [f"percentile_{m}" for m in method_ids]
        + ["consensus_score", "n_methods_at_N", "methods_that_selected_it_at_N",
           "selection_rank"]
    )
    return {
        "candidate_table": table,
        "selected": selected,
        "pool": pool,
        "missing_ranks": missing,
        "extra_cols": extras,
    }


def quota_at_n(rankings, topn_sets, method_ids, n):
    """Deterministic round-robin over each method's own top-N safety set.

    A method may only propose neurons from its existing top-N file, ordered
    by that method's full ranking. It never walks further down the ranking
    once that set is exhausted. Duplicates are skipped and the same method
    immediately proposes its next unused top-N neuron.
    """
    lists = {}
    for mid in method_ids:
        lists[mid] = [pair for pair, _rank in ordered_topn(rankings[mid], topn_sets[mid])]

    maps = {
        mid: {(int(l), int(i)): int(r)
              for l, i, r in zip(rankings[mid]["layer"], rankings[mid]["neuron_index"],
                                 rankings[mid]["rank"])}
        for mid in method_ids
    }
    n_ranked = {mid: len(rankings[mid]) for mid in method_ids}

    selected = []
    seen = set()
    contributed = Counter()
    skipped_duplicates = 0
    exhausted = []
    pointers = {mid: 0 for mid in method_ids}
    active = list(method_ids)
    step = 0

    while len(selected) < n:
        if not active:
            raise RuntimeError(
                f"quota exhausted every method's top-N set after {len(selected)} "
                f"neurons; cannot reach N={n}"
            )
        progressed = False
        still_active = []
        for mid in active:
            if len(selected) >= n:
                still_active.append(mid)
                continue
            lst = lists[mid]
            picked = None
            while pointers[mid] < len(lst):
                pair = lst[pointers[mid]]
                pointers[mid] += 1
                if pair in seen:
                    skipped_duplicates += 1
                    continue
                picked = pair
                break
            if picked is None:
                if mid not in exhausted:
                    exhausted.append(mid)
                continue
            still_active.append(mid)
            step += 1
            seen.add(picked)
            contributed[mid] += 1
            others = [m for m in method_ids if m != mid and picked in topn_sets[m]]
            row = {
                "variant": "quota",
                "layer": picked[0],
                "neuron_index": picked[1],
                "entered_via": mid,
                "also_in": ";".join(others),
                "n_methods_at_N": 1 + len(others),
                "selection_step": step,
            }
            percentiles = []
            for m in method_ids:
                r = maps[m].get(picked, n_ranked[m] + 1)
                row[f"rank_{m}"] = r
                p = percentile(r, n_ranked[m])
                row[f"percentile_{m}"] = p
                percentiles.append(p)
            row["consensus_score"] = float(np.mean(percentiles))
            selected.append(row)
            progressed = True
        active = still_active
        if not progressed and len(selected) < n:
            raise RuntimeError(
                f"quota made no progress at size {len(selected)}; N={n}"
            )

    extra_cols = (
        ["entered_via", "also_in", "n_methods_at_N", "selection_step"]
        + [f"rank_{m}" for m in method_ids]
        + [f"percentile_{m}" for m in method_ids]
        + ["consensus_score"]
    )
    return {
        "selected": pd.DataFrame(selected),
        "contributed": dict(contributed),
        "skipped_duplicates": skipped_duplicates,
        "exhausted": exhausted,
        "pool": candidate_pool(topn_sets),
        "extra_cols": extra_cols,
        "lists": lists,
    }


def random_pairs(n, universe_size, intermediate_size, seed):
    """Repo B random_{n}: torch.randperm(universe, seed)[:n] sorted.

    Reproduced from safety-neurons-where-you-look
    `src/build_ablation_pilot_targets.py` (RANDOM1_SEED = 1000).
    """
    import torch

    rng = torch.Generator().manual_seed(int(seed))
    flat = torch.randperm(int(universe_size), generator=rng)[: int(n)].sort().values
    pairs = []
    for v in flat.tolist():
        layer, idx = divmod(int(v), int(intermediate_size))
        pairs.append((layer, idx))
    return pairs, [int(v) for v in flat.tolist()]


def composition_report(cfg, n, method_ids, topn_sets, consensus_info, quota_info,
                       random_pairs_set, mismatch_methods=None, num_layers=None):
    pool = consensus_info["pool"]
    cons = set(zip(consensus_info["selected"]["layer"].astype(int),
                   consensus_info["selected"]["neuron_index"].astype(int)))
    quot = set(zip(quota_info["selected"]["layer"].astype(int),
                   quota_info["selected"]["neuron_index"].astype(int)))

    def jaccard(a, b):
        u = a | b
        return (len(a & b) / len(u)) if u else 1.0

    rows = []
    n_layers = num_layers if num_layers is not None else cfg["num_layers"]
    rows.append(_row("candidate_pool", n, pool, method_ids, topn_sets, n_layers, extra={
        "note": "union of the existing top-N safety-neuron files",
    }))
    rows.append(_row("rank_consensus", n, cons, method_ids, topn_sets, n_layers, extra={
        "jaccard_vs_quota": jaccard(cons, quot),
        "overlap_with_quota": len(cons & quot),
    }))
    rows.append(_row("quota", n, quot, method_ids, topn_sets, n_layers, extra={
        "jaccard_vs_consensus": jaccard(quot, cons),
        "overlap_with_consensus": len(quot & cons),
        "skipped_duplicates": quota_info["skipped_duplicates"],
        "exhausted_methods": ";".join(quota_info["exhausted"]),
        **{f"contributed_{m}": quota_info["contributed"].get(m, 0) for m in method_ids},
    }))
    rows.append(_row("random", n, random_pairs_set, method_ids, topn_sets, n_layers, extra={
        "note": f"Repo B random_{{n}} with seed {cfg['random_seed']}",
    }))
    for mid in method_ids:
        extra = {}
        if mismatch_methods and mid in mismatch_methods:
            extra["ranking_vs_topn"] = "MISMATCH"
        rows.append(_row(mid, n, topn_sets[mid], method_ids, topn_sets, n_layers, extra=extra))

    # Consensus multiplicity histogram: how many selected neurons sit in
    # 1, 2, ..., K individual top-N sets.
    hist = Counter(int(v) for v in consensus_info["selected"]["n_methods_at_N"])
    rows.append({
        "selection": "rank_consensus_multiplicity",
        "N": n,
        "size": len(cons),
        **{f"in_{k}_topN_sets": hist.get(k, 0) for k in range(1, len(method_ids) + 1)},
    })
    return pd.DataFrame(rows)


def _row(name, n, pairs, method_ids, topn_sets, num_layers, extra=None):
    layer_counts = Counter(l for l, _ in pairs)
    row = {
        "selection": name,
        "N": n,
        "size": len(pairs),
        "candidate_pool_size": len(candidate_pool(topn_sets)),
    }
    for mid in method_ids:
        inter = len(pairs & topn_sets[mid])
        row[f"overlap_{mid}"] = inter
        row[f"pct_from_{mid}"] = (100.0 * inter / len(pairs)) if pairs else 0.0
        row[f"jaccard_{mid}"] = _jaccard(pairs, topn_sets[mid])
    for layer in range(num_layers):
        row[f"layer_{layer}"] = layer_counts.get(layer, 0)
    if extra:
        row.update(extra)
    return row


def _jaccard(a, b):
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def save_target_json(path, name, pairs, intermediate_size, category, extra=None):
    import json
    from pathlib import Path

    flat = sorted(int(l) * intermediate_size + int(i) for l, i in pairs)
    entry = {
        "name": name,
        "category": category,
        "n_neurons": len(flat),
        "flat_indices": flat,
    }
    if extra:
        entry.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f)
    return {k: v for k, v in entry.items() if k != "flat_indices"}

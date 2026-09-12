"""Load full rankings and top-N safety-neuron files into a common schema.

Every neuron is identified as (layer, neuron_index). Rank 1 is treated as
the highest-ranked neuron. Formats differ across methods; column names are
read from methods.yaml, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import resolve

NEURON_COLS = ("layer", "neuron_index")


def neuron_id(layer, neuron_index):
    return (int(layer), int(neuron_index))


def to_pairs(df):
    return list(zip(df["layer"].astype(int).tolist(),
                    df["neuron_index"].astype(int).tolist()))


def as_set(pairs):
    return set(neuron_id(l, i) for l, i in pairs)


def ranking_path(cfg, method_id):
    spec = cfg["method_by_id"][method_id]
    return resolve(cfg, spec["ranking"]["path"])


def topn_path(cfg, method_id, n):
    spec = cfg["method_by_id"][method_id]
    return resolve(cfg, spec["topn"]["path"].format(n=int(n)))


def load_ranking(cfg, method_id):
    """Return a DataFrame sorted by rank ascending (1 = best).

    Columns: layer, neuron_index, rank, and score if a score column exists.
    """
    spec = cfg["method_by_id"][method_id]["ranking"]
    path = resolve(cfg, spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"{method_id}: ranking missing: {path}")
    raw = pd.read_csv(path)
    layer_c = spec["layer_column"]
    neuron_c = spec["neuron_column"]
    rank_c = spec.get("rank_column")
    score_c = spec.get("score_column")
    for col in (layer_c, neuron_c):
        if col not in raw.columns:
            raise ValueError(f"{method_id}: {path.name} has no column {col!r}; "
                             f"columns={list(raw.columns)}")
    out = pd.DataFrame({
        "layer": raw[layer_c].astype(np.int32),
        "neuron_index": raw[neuron_c].astype(np.int32),
    })
    if rank_c:
        if rank_c not in raw.columns:
            raise ValueError(f"{method_id}: {path.name} has no rank column {rank_c!r}")
        out["rank"] = raw[rank_c].astype(np.int32)
    else:
        out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    if score_c and score_c in raw.columns:
        out["score"] = raw[score_c].astype(np.float64)
    else:
        out["score"] = np.nan
    out = out.sort_values("rank", kind="mergesort").reset_index(drop=True)
    return out


def load_all_rankings(cfg, methods=None):
    methods = methods or list(cfg["method_ids"])
    return {mid: load_ranking(cfg, mid) for mid in methods}


def rank_maps(rankings):
    """method_id -> {(layer, neuron_index): rank}."""
    maps = {}
    for mid, df in rankings.items():
        maps[mid] = {
            (int(l), int(i)): int(r)
            for l, i, r in zip(df["layer"], df["neuron_index"], df["rank"])
        }
    return maps


def load_topn(cfg, method_id, n):
    """Return a set of (layer, neuron_index) from the existing top-N file."""
    spec = cfg["method_by_id"][method_id]["topn"]
    path = topn_path(cfg, method_id, n)
    if not path.exists():
        raise FileNotFoundError(f"{method_id}: top-N file missing for N={n}: {path}")
    kind = spec.get("kind", "csv")
    if kind == "layer_json":
        return _load_layer_json(path), path
    if kind == "csv":
        return _load_csv_pairs(path), path
    raise ValueError(f"{method_id}: unknown topn.kind {kind!r}")


def _load_layer_json(path):
    with open(path) as f:
        raw = json.load(f)
    out = set()
    for key, values in raw.items():
        if not key.startswith("layer"):
            continue
        layer = int(key[len("layer"):])
        for idx in values:
            out.add((layer, int(idx)))
    return out


def _load_csv_pairs(path):
    df = pd.read_csv(path)
    for col in ("layer", "neuron_index"):
        if col not in df.columns:
            raise ValueError(f"{path} missing {col}; columns={list(df.columns)}")
    return set(zip(df["layer"].astype(int).tolist(),
                   df["neuron_index"].astype(int).tolist()))


def load_all_topn(cfg, n, methods=None):
    methods = methods or list(cfg["method_ids"])
    sets, paths = {}, {}
    for mid in methods:
        s, p = load_topn(cfg, mid, n)
        sets[mid] = s
        paths[mid] = p
    return sets, paths


def candidate_pool(topn_sets):
    """C_N = union of every method's existing top-N safety-neuron set."""
    pool = set()
    for s in topn_sets.values():
        pool |= s
    return pool


def methods_selecting(topn_sets, pair):
    return [mid for mid, s in topn_sets.items() if pair in s]


def ordered_topn(ranking_df, topn_set):
    """Neurons in `topn_set`, ordered by this method's full-ranking rank.

    Neurons that appear in the top-N file but not in the ranking are
    appended at the end (worst) and should already have been flagged by
    validation.
    """
    ranked = []
    seen = set()
    for layer, idx, rank in zip(ranking_df["layer"], ranking_df["neuron_index"],
                                ranking_df["rank"]):
        pair = (int(layer), int(idx))
        if pair in topn_set:
            ranked.append((pair, int(rank)))
            seen.add(pair)
    missing = sorted(topn_set - seen)
    for pair in missing:
        ranked.append((pair, None))
    return ranked


def percentile(rank, n_ranked):
    if n_ranked <= 1:
        return 0.0
    return (int(rank) - 1) / (n_ranked - 1)


def write_selection_csv(path, rows, extra_cols=None):
    """Write a neuron-selection CSV. Always includes layer, neuron_index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    front = ["variant", "layer", "neuron_index"]
    extra_cols = extra_cols or []
    cols = [c for c in front if c in df.columns]
    cols += [c for c in extra_cols if c in df.columns and c not in cols]
    cols += [c for c in df.columns if c not in cols]
    df[cols].to_csv(path, index=False)
    return path


def selection_from_pairs(pairs):
    """dict[layer] -> sorted unique neuron-index array."""
    by_layer = {}
    for layer, idx in pairs:
        by_layer.setdefault(int(layer), []).append(int(idx))
    return {l: np.unique(np.asarray(v, dtype=np.int64)) for l, v in by_layer.items()}


def pairs_from_selection(sel):
    pairs = []
    for layer in sorted(sel):
        for idx in sel[layer]:
            pairs.append((int(layer), int(idx)))
    return pairs


def flat_indices(pairs, intermediate_size):
    return [int(layer) * intermediate_size + int(idx) for layer, idx in pairs]


def pairs_from_flat(flat, intermediate_size):
    out = []
    for v in flat:
        layer, idx = divmod(int(v), intermediate_size)
        out.append((layer, idx))
    return out


def load_selection_csv(path):
    df = pd.read_csv(path)
    return as_set(zip(df["layer"].astype(int), df["neuron_index"].astype(int))), df

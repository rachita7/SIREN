"""CPU smoke tests for rank_ensemble (no GPU, no model weights).

    python rank_ensemble/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import json

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregators import (  # noqa: E402
    quota_at_n,
    random_pairs,
    rank_consensus_at_n,
)
from config import load_config  # noqa: E402
from load_rankings import (  # noqa: E402
    candidate_pool,
    load_all_rankings,
    load_all_topn,
    percentile,
)
from refusal_labels import CHECK_PREFIX_CHARS, is_refusal  # noqa: E402
from validate_rankings import run_validation  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def write_synthetic(tmp):
    """3 methods, 4 layers x 5 neurons = 20. Rankings agree with top-N."""
    tmp = Path(tmp)
    (tmp / "rank").mkdir()
    (tmp / "topn").mkdir()
    n_layers, width = 4, 5
    universe = [(l, i) for l in range(n_layers) for i in range(width)]

    # method_a: identity order
    # method_b: reverse order
    # method_c: even neurons first, then odds
    orders = {
        "method_a": universe[:],
        "method_b": list(reversed(universe)),
        "method_c": [p for p in universe if p[1] % 2 == 0] + [p for p in universe if p[1] % 2 == 1],
    }
    for mid, order in orders.items():
        rows = [{"rank": r + 1, "layer": l, "neuron_index": i, "score": 100 - r}
                for r, (l, i) in enumerate(order)]
        pd.DataFrame(rows).to_csv(tmp / "rank" / f"{mid}.csv", index=False)
        for n in (4, 8, 12):
            top = order[:n]
            pd.DataFrame(
                {"variant": mid, "layer": [p[0] for p in top],
                 "neuron_index": [p[1] for p in top]}
            ).to_csv(tmp / "topn" / f"{mid}_N{n}.csv", index=False)

    cfg = {
        "num_layers": n_layers,
        "intermediate_size": width,
        "default_budgets": [4, 8, 12],
        "eval_seed": 4242,
        "random_seed": 1000,
        "model": {"name": "dummy"},
        "methods": [
            {
                "id": mid,
                "display": mid,
                "ranking": {
                    "path": str(tmp / "rank" / f"{mid}.csv"),
                    "rank_column": "rank",
                    "layer_column": "layer",
                    "neuron_column": "neuron_index",
                    "score_column": "score",
                },
                "topn": {
                    "path": str(tmp / "topn" / f"{mid}_N{{n}}.csv"),
                    "kind": "csv",
                },
            }
            for mid in orders
        ],
    }
    cfg_path = tmp / "methods.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path, orders, n_layers, width


def test_synthetic():
    print("synthetic universe")
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path, orders, n_layers, width = write_synthetic(tmp)
        cfg = load_config(cfg_path)
        methods = cfg["method_ids"]
        rankings = load_all_rankings(cfg, methods)

        for mid, df in rankings.items():
            pairs = list(zip(df.layer.astype(int), df.neuron_index.astype(int)))
            check(f"{mid}: unique neurons", len(pairs) == len(set(pairs)))
            check(f"{mid}: ranks 1..M", df["rank"].min() == 1 and df["rank"].nunique() == len(df))
            check(f"{mid}: ids in range",
                  df.layer.min() >= 0 and df.layer.max() < n_layers
                  and df.neuron_index.min() >= 0 and df.neuron_index.max() < width)

        val = run_validation(cfg, [4, 8, 12], methods)
        check("synthetic ranking prefix matches top-N", val["ok"])

        for n in (4, 8, 12):
            topn_sets, _ = load_all_topn(cfg, n, methods)
            pool = candidate_pool(topn_sets)
            check(f"N={n}: every C_N neuron is in at least one top-N",
                  all(any(p in s for s in topn_sets.values()) for p in pool))
            cons = rank_consensus_at_n(rankings, topn_sets, methods, n)
            sel = list(zip(cons["selected"].layer.astype(int),
                           cons["selected"].neuron_index.astype(int)))
            check(f"N={n}: consensus has exactly N unique",
                  len(sel) == n and len(set(sel)) == n)
            check(f"N={n}: consensus ⊆ C_N", set(sel) <= pool)
            check(f"N={n}: consensus sorted by score",
                  cons["selected"]["consensus_score"].is_monotonic_increasing)

            quot = quota_at_n(rankings, topn_sets, methods, n)
            qsel = list(zip(quot["selected"].layer.astype(int),
                            quot["selected"].neuron_index.astype(int)))
            check(f"N={n}: quota has exactly N unique",
                  len(qsel) == n and len(set(qsel)) == n)
            check(f"N={n}: quota ⊆ C_N", set(qsel) <= pool)
            for _, row in quot["selected"].iterrows():
                src = row["entered_via"]
                pair = (int(row["layer"]), int(row["neuron_index"]))
                check(f"N={n}: quota {pair} came from {src} top-N",
                      pair in topn_sets[src])

            cons2 = rank_consensus_at_n(rankings, topn_sets, methods, n)
            quot2 = quota_at_n(rankings, topn_sets, methods, n)
            check(f"N={n}: consensus deterministic",
                  cons["selected"].equals(cons2["selected"]))
            check(f"N={n}: quota deterministic",
                  quot["selected"].equals(quot2["selected"]))

        # percentile formula
        check("percentile rank 1 is 0", percentile(1, 20) == 0.0)
        check("percentile rank M is 1", percentile(20, 20) == 1.0)
        check("percentile midpoint", abs(percentile(11, 21) - 0.5) < 1e-12)


def test_refusal():
    print("refusal_labels (Repo B)")
    check("canonical refusal", is_refusal("I can't help with that."))
    check("prefix window", is_refusal("I cannot" + " x" * 20))
    late = ("Sure, here is the recipe. " * 20) + "I can't guarantee anything."
    check("late hedge is not a refusal", not is_refusal(late))
    check("prefix length", CHECK_PREFIX_CHARS == 200)


def test_hook():
    print("ablation hook")
    from ablation import make_zero_input_hook

    x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    idx = torch.tensor([1, 4])
    hook = make_zero_input_hook(idx)
    class Dummy:
        pass
    out = hook(Dummy(), (x, "other"))
    check("hook returns tuple", isinstance(out, tuple) and out[1] == "other")
    check("selected channels zeroed",
          torch.equal(out[0][..., [1, 4]], torch.zeros_like(out[0][..., [1, 4]])))
    check("other channels unchanged",
          torch.equal(out[0][..., [0, 2, 3, 5]], x[..., [0, 2, 3, 5]]))
    check("input not mutated", not torch.equal(x[..., 1], torch.zeros_like(x[..., 1])))


def test_random_repo_b():
    print("Repo B random control")
    a, fa = random_pairs(20, 80, 5, 1000)
    b, fb = random_pairs(20, 80, 5, 1000)
    c, _ = random_pairs(20, 80, 5, 1001)
    check("same seed -> same random set", a == b and fa == fb)
    check("different seed -> different set", set(a) != set(c))
    check("random unique and size N", len(a) == 20 and len(set(a)) == 20)
    check("random indices in universe",
          all(0 <= l < 16 and 0 <= i < 5 for l, i in a))


def test_real_data():
    print("real Repo A rankings")
    cfg_path = HERE / "methods.json"
    if not cfg_path.exists():
        check("methods.json present", False)
        return
    cfg = load_config(cfg_path)
    methods = cfg["method_ids"]
    check("config lists the configured methods", len(methods) >= 2)
    rankings = load_all_rankings(cfg, methods)
    expected = cfg["universe_size"]
    universes = []
    for mid, df in rankings.items():
        pairs = set(zip(df.layer.astype(int), df.neuron_index.astype(int)))
        universes.append(pairs)
        check(f"{mid}: full universe", len(df) == expected and len(pairs) == expected)
        check(f"{mid}: no duplicate neurons", len(pairs) == len(df))
        check(f"{mid}: valid layer/index",
              int(df.layer.min()) >= 0 and int(df.layer.max()) < cfg["num_layers"]
              and int(df.neuron_index.min()) >= 0
              and int(df.neuron_index.max()) < cfg["intermediate_size"])
        check(f"{mid}: rank 1 is present", int(df["rank"].min()) == 1)
    check("all methods share one universe", all(u == universes[0] for u in universes))

    val = run_validation(cfg, cfg["default_budgets"], methods)
    matches = {r["method"] for r in val["prefix_rows"] if r["ok"]}
    mismatches = sorted({r["method"] for r in val["prefix_rows"] if not r["ok"]})
    for mid in methods:
        check(f"{mid}: ranking prefix == published top-N", mid in matches)
    check("no ranking-vs-top-N mismatches", not mismatches, str(mismatches))
    check("no identical ranking files", not val["identical_rankings"],
          str(val["identical_rankings"]))
    check("validation passes", val["ok"])

    # Build one small real ensemble (N=459) and enforce the experimental rule.
    n = 459
    if n in cfg["default_budgets"]:
        topn_sets, _ = load_all_topn(cfg, n, methods)
        pool = candidate_pool(topn_sets)
        cons = rank_consensus_at_n(rankings, topn_sets, methods, n)
        quot = quota_at_n(rankings, topn_sets, methods, n)
        cset = set(zip(cons["selected"].layer.astype(int),
                       cons["selected"].neuron_index.astype(int)))
        qset = set(zip(quot["selected"].layer.astype(int),
                       quot["selected"].neuron_index.astype(int)))
        check("real consensus: exactly N unique", len(cset) == n)
        check("real quota: exactly N unique", len(qset) == n)
        check("real consensus ⊆ C_N", cset <= pool)
        check("real quota ⊆ C_N", qset <= pool)
        for _, row in quot["selected"].iterrows():
            if (int(row.layer), int(row.neuron_index)) not in topn_sets[row.entered_via]:
                check("quota never leaves its own top-N", False)
                break
        else:
            check("quota never leaves its own top-N", True)


def main():
    test_synthetic()
    test_refusal()
    test_hook()
    test_random_repo_b()
    test_real_data()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for name in FAILS:
            print(f"  - {name}")
        raise SystemExit(1)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()

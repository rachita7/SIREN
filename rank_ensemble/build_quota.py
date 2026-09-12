"""Build the quota / round-robin ensemble at one or more budgets.

    python rank_ensemble/build_quota.py --budgets 459 2294 4588 9175
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregators import quota_at_n
from config import HERE as CFG_HERE, load_config, method_ids
from load_rankings import load_all_rankings, load_all_topn, write_selection_csv


def build_one(cfg, rankings, n, methods, out_dir):
    topn_sets, _ = load_all_topn(cfg, n, methods)
    info = quota_at_n(rankings, topn_sets, methods, n)
    sel_path = out_dir / f"quota_N{n}.csv"
    write_selection_csv(sel_path, info["selected"], extra_cols=info["extra_cols"])
    return info, sel_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods = method_ids(cfg, args.methods)
    budgets = args.budgets or cfg["default_budgets"]
    out_dir = Path(args.output_dir) if args.output_dir else CFG_HERE / "selections"
    rankings = load_all_rankings(cfg, methods)
    for n in budgets:
        info, sel_path = build_one(cfg, rankings, n, methods, out_dir)
        contrib = " ".join(f"{m}={info['contributed'].get(m, 0)}" for m in methods)
        print(f"N={n}: selected={len(info['selected'])} skipped_dupes="
              f"{info['skipped_duplicates']} exhausted={info['exhausted'] or 'none'} "
              f"contrib[{contrib}] -> {sel_path.name}")


if __name__ == "__main__":
    main()

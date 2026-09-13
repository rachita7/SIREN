"""Build consensus, quota, and random selections for every requested budget.

Runs validation first. If a full ranking does not reproduce the published
top-N safety-neuron file, the script stops.

    python rank_ensemble/build_ensembles.py
    python rank_ensemble/build_ensembles.py --budgets 459 2294 4588 9175
    python rank_ensemble/build_ensembles.py --experiment siren_yang_wang_zhao
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregators import (
    composition_report,
    consensus_table,
    quota_at_n,
    random_pairs,
    rank_consensus_at_n,
    save_target_json,
)
from config import EXPERIMENTS, display_name, load_config, method_ids, resolve_experiment
from load_rankings import (
    load_all_rankings,
    load_all_topn,
    write_selection_csv,
)
from validate_rankings import run_validation


def _mismatch_methods(prefix_rows):
    return sorted({r["method"] for r in prefix_rows if not r["ok"]})


def build_budget(cfg, rankings, n, methods, selections_dir, results_dir,
                 write_global_consensus, mismatch_methods):
    topn_sets, _ = load_all_topn(cfg, n, methods)
    cons = rank_consensus_at_n(rankings, topn_sets, methods, n)
    quot = quota_at_n(rankings, topn_sets, methods, n)
    rand_pairs, rand_flat = random_pairs(
        n, cfg["universe_size"], cfg["intermediate_size"], cfg["random_seed"]
    )
    rand_set = set(rand_pairs)
    rand_df = {
        "variant": ["random"] * n,
        "layer": [p[0] for p in rand_pairs],
        "neuron_index": [p[1] for p in rand_pairs],
        "flat_index": rand_flat,
        "seed": [cfg["random_seed"]] * n,
    }

    write_selection_csv(selections_dir / f"candidate_pool_N{n}.csv",
                        cons["candidate_table"])
    write_selection_csv(selections_dir / f"rank_consensus_N{n}.csv",
                        cons["selected"], extra_cols=cons["extra_cols"])
    write_selection_csv(selections_dir / f"quota_N{n}.csv",
                        quot["selected"], extra_cols=quot["extra_cols"])
    write_selection_csv(selections_dir / f"random_N{n}.csv", rand_df)

    if write_global_consensus:
        all_pairs = list(zip(rankings[methods[0]]["layer"].astype(int),
                             rankings[methods[0]]["neuron_index"].astype(int)))
        global_df, _ = consensus_table(rankings, all_pairs, methods)
        write_selection_csv(results_dir / "consensus_scores_universe.csv", global_df)

    comp = composition_report(
        cfg, n, methods, topn_sets, cons, quot, rand_set,
        mismatch_methods=set(mismatch_methods),
        num_layers=cfg["num_layers"],
    )
    comp_path = results_dir / f"composition_N{n}.csv"
    comp.to_csv(comp_path, index=False)

    targets_dir = results_dir / f"targets_N{n}"
    manifest = []
    width = cfg["intermediate_size"]
    manifest.append(save_target_json(
        targets_dir / "baseline_no_ablation.json", "baseline_no_ablation",
        [], width, "baseline",
    ))
    for mid in methods:
        manifest.append(save_target_json(
            targets_dir / f"{mid}.json", mid, topn_sets[mid], width,
            "individual", extra={"display": display_name(cfg, mid)},
        ))
    cons_pairs = list(zip(cons["selected"]["layer"].astype(int),
                          cons["selected"]["neuron_index"].astype(int)))
    quot_pairs = list(zip(quot["selected"]["layer"].astype(int),
                          quot["selected"]["neuron_index"].astype(int)))
    manifest.append(save_target_json(
        targets_dir / "rank_consensus.json", "rank_consensus", cons_pairs, width,
        "ensemble",
    ))
    manifest.append(save_target_json(
        targets_dir / "quota.json", "quota", quot_pairs, width, "ensemble",
        extra={
            "skipped_duplicates": quot["skipped_duplicates"],
            "exhausted_methods": quot["exhausted"],
            "contributed": quot["contributed"],
        },
    ))
    manifest.append(save_target_json(
        targets_dir / "random.json", "random", rand_pairs, width, "random",
        extra={"seed": cfg["random_seed"],
               "source": "Repo B build_ablation_pilot_targets.random_n"},
    ))
    with open(targets_dir / "manifest.json", "w") as f:
        json.dump({
            "n_target": n,
            "num_layers": cfg["num_layers"],
            "intermediate_size": width,
            "methods": methods,
            "targets": manifest,
        }, f, indent=2)

    # Sanity: ensembles are exactly N unique neurons, all inside C_N.
    assert len(set(cons_pairs)) == n
    assert len(set(quot_pairs)) == n
    assert set(cons_pairs) <= cons["pool"]
    assert set(quot_pairs) <= cons["pool"]
    for mid, lst in quot["lists"].items():
        assert set(lst) <= topn_sets[mid]

    return {
        "N": n,
        "pool_size": len(cons["pool"]),
        "consensus": len(cons_pairs),
        "quota": len(quot_pairs),
        "quota_skipped_duplicates": quot["skipped_duplicates"],
        "quota_exhausted": quot["exhausted"],
        "quota_contributed": quot["contributed"],
        "composition_path": str(comp_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="all7", choices=list(EXPERIMENTS),
                        help="Named method subset + output folder. "
                             "all7 writes to rank_ensemble/{selections,results}; "
                             "other names write under experiments/<name>/")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Neuron budgets (default: values in methods.json)")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--selections-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--write-global-consensus", action="store_true",
                        help="Also write results/consensus_scores_universe.csv "
                             "(~100 MB; diagnostic only, not used for selection).")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = resolve_experiment(args.experiment)
    methods = method_ids(cfg, args.methods or exp["methods"])
    budgets = args.budgets or cfg["default_budgets"]
    selections_dir = Path(args.selections_dir) if args.selections_dir else exp["selections_dir"]
    results_dir = Path(args.results_dir) if args.results_dir else exp["results_dir"]
    print(f"experiment={exp['name']} ({exp['display']})")
    print(f"methods={methods}")
    print(f"selections_dir={selections_dir}")
    print(f"results_dir={results_dir}")
    selections_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mismatch_methods = []
    if not args.skip_validation:
        val = run_validation(cfg, budgets, methods)
        report_path = results_dir / "validation_report.md"
        report_path.write_text(val["report"])
        print(val["report"])
        print(f"Wrote {report_path}\n")
        mismatch_methods = _mismatch_methods(val["prefix_rows"])
        if not val["ok"]:
            raise SystemExit(
                "Validation failed. Read the report above and fix the "
                "ranking or top-N files before building ensembles."
            )

    rankings = load_all_rankings(cfg, methods)
    write_global = args.write_global_consensus
    summaries = []
    for i, n in enumerate(budgets):
        print(f"Building ensembles at N={n} ...")
        summary = build_budget(
            cfg, rankings, n, methods, selections_dir, results_dir,
            write_global_consensus=(write_global and i == 0),
            mismatch_methods=mismatch_methods,
        )
        summaries.append(summary)
        print(f"  |C_N|={summary['pool_size']}  consensus={summary['consensus']}  "
              f"quota={summary['quota']}  quota skipped dupes="
              f"{summary['quota_skipped_duplicates']}  exhausted="
              f"{summary['quota_exhausted'] or 'none'}")
        write_global = False

    print("\nDone.")
    print(f"Selections: {selections_dir}")
    print(f"Targets:    {results_dir}/targets_N<N>/")
    print(f"Composition:{results_dir}/composition_N<N>.csv")


if __name__ == "__main__":
    main()

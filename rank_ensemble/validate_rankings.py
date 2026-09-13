"""Validate full rankings against each other and against existing top-N files.

The experimental contract is:

  * top-N safety-neuron files define eligibility
  * full-ranking files provide exact ranks

Those two sources MUST be checked against each other. If the first N
neurons of a full ranking do not reproduce that method's published top-N
set, this script reports the discrepancy and exits non-zero. It does not
silently pick a winner.

    python rank_ensemble/validate_rankings.py
    python rank_ensemble/validate_rankings.py --budgets 459 2294 --warn-only
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from config import HERE, display_name, load_config, method_ids
from load_rankings import load_all_rankings, load_all_topn, ranking_path, topn_path


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_ranking(cfg, method_id, df):
    n_layers = cfg["num_layers"]
    width = cfg["intermediate_size"]
    expected = cfg["universe_size"]
    pairs = list(zip(df["layer"].astype(int), df["neuron_index"].astype(int)))
    n = len(df)
    n_unique = len(set(pairs))
    ranks = df["rank"].to_numpy()
    issues = []
    if n != n_unique:
        issues.append(f"duplicate neurons: {n - n_unique}")
    if ranks.min() != 1 or ranks.max() != n or len(set(ranks.tolist())) != n:
        issues.append(
            f"ranks are not unique 1..{n} (min={ranks.min()}, max={ranks.max()}, "
            f"unique={len(set(ranks.tolist()))})"
        )
    if (df["layer"] < 0).any() or (df["layer"] >= n_layers).any():
        issues.append(f"layer outside 0..{n_layers - 1}")
    if (df["neuron_index"] < 0).any() or (df["neuron_index"] >= width).any():
        issues.append(f"neuron_index outside 0..{width - 1}")
    if n != expected:
        issues.append(f"ranked {n} neurons; expected universe {expected}")

    scores = df["score"].to_numpy()
    finite = np.isfinite(scores)
    n_tied_values = 0
    n_tied_neurons = 0
    if finite.any():
        vc = Counter(scores[finite].tolist())
        tied = {s: c for s, c in vc.items() if c > 1}
        n_tied_values = len(tied)
        n_tied_neurons = sum(tied.values())

    return {
        "method": method_id,
        "display": display_name(cfg, method_id),
        "n": n,
        "n_unique": n_unique,
        "rank_1_is_best": True,
        "duplicate_neurons": n - n_unique,
        "tied_score_values": n_tied_values,
        "tied_score_neurons": n_tied_neurons,
        "score_monotonic_nonincreasing": bool(
            finite.all() and np.all(scores[1:] <= scores[:-1] + 1e-15)
        ),
        "universe_ok": n == expected and n == n_unique,
        "issues": issues,
        "path": str(ranking_path(cfg, method_id)),
        "md5": _md5(ranking_path(cfg, method_id)),
        "pairs": set(pairs),
    }


def compare_universes(inspections):
    rows = []
    ids = [r["method"] for r in inspections]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            sa, sb = inspections[i]["pairs"], None
            for r in inspections:
                if r["method"] == b:
                    sb = r["pairs"]
                    break
            sa = inspections[i]["pairs"]
            rows.append({
                "a": a, "b": b,
                "intersection": len(sa & sb),
                "only_a": len(sa - sb),
                "only_b": len(sb - sa),
                "same": sa == sb,
            })
    return rows


def compare_prefix_to_topn(cfg, rankings, budgets, methods):
    rows = []
    for mid in methods:
        df = rankings[mid]
        prefix_pairs = list(zip(df["layer"].astype(int), df["neuron_index"].astype(int)))
        for n in budgets:
            try:
                topn_sets, paths = load_all_topn(cfg, n, methods=[mid])
            except FileNotFoundError as exc:
                rows.append({
                    "method": mid, "N": n, "ok": False, "missing_file": True,
                    "error": str(exc),
                    "rank_n": 0, "topn_n": 0, "size_offset": 0,
                    "intersection": 0,
                    "only_rank": 0, "only_topn": 0, "jaccard": 0.0,
                    "path": str(topn_path(cfg, mid, n)),
                })
                continue
            topn = topn_sets[mid]
            # Compare the ranking prefix of length |top-N file|. When the
            # published file is a few neurons off N (ties at the cutoff),
            # that file is still a prefix of the ranking.
            prefix = set(prefix_pairs[:len(topn)])
            inter = prefix & topn
            union = prefix | topn
            jac = len(inter) / len(union) if union else 1.0
            rows.append({
                "method": mid, "N": n, "ok": prefix == topn,
                "missing_file": False, "error": "",
                "rank_n": len(prefix), "topn_n": len(topn),
                "size_offset": len(topn) - n,
                "intersection": len(inter),
                "only_rank": len(prefix - topn),
                "only_topn": len(topn - prefix),
                "jaccard": jac,
                "path": str(paths[mid]),
            })
    return rows


def identical_ranking_files(inspections):
    by_md5 = {}
    for row in inspections:
        by_md5.setdefault(row["md5"], []).append(row["method"])
    return [ids for ids in by_md5.values() if len(ids) > 1]


def render_report(cfg, inspections, universe_pairs, prefix_rows, identical, budgets):
    lines = []
    lines.append("# Rank-ensemble validation report")
    lines.append("")
    lines.append("Full rankings supply exact ranks. Existing top-N safety-neuron")
    lines.append("files supply eligibility. This report checks both, and checks")
    lines.append("whether the first N neurons of each ranking reproduce that")
    lines.append("method's published top-N set.")
    lines.append("")
    lines.append(f"- Config: `{cfg['path']}`")
    lines.append(f"- Methods: {', '.join(m['method'] for m in inspections)}")
    lines.append(f"- Expected neuron universe: {cfg['num_layers']} layers × "
                 f"{cfg['intermediate_size']} = {cfg['universe_size']}")
    lines.append(f"- Budgets checked: {budgets}")
    lines.append("- Rank convention: **1 = highest-ranked**")
    lines.append("")

    lines.append("## 1. Full-ranking integrity")
    lines.append("")
    lines.append("| Method | N ranked | Unique | Dup neurons | Tied scores "
                 "(values / neurons) | Score monotone ↓ | Issues |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in inspections:
        issues = "; ".join(row["issues"]) if row["issues"] else "none"
        lines.append(
            f"| {row['display']} (`{row['method']}`) | {row['n']} | {row['n_unique']} | "
            f"{row['duplicate_neurons']} | {row['tied_score_values']} / {row['tied_score_neurons']} | "
            f"{'yes' if row['score_monotonic_nonincreasing'] else 'no'} | {issues} |"
        )
    lines.append("")
    lines.append("Score ties are recorded here. Every ranking already assigns a")
    lines.append("unique integer rank, so percentile ranks use that assigned rank")
    lines.append("rather than re-breaking ties.")
    lines.append("")
    lines.append("SIREN's `global_rank` follows per-layer entry-threshold order,")
    lines.append("not a global sort of `abs_weight`. That is why SIREN's score")
    lines.append("column is not monotone in rank. Rank 1 is still the official")
    lines.append("highest-ranked neuron.")
    lines.append("")

    lines.append("## 2. Shared neuron universe")
    lines.append("")
    if universe_pairs and all(r["same"] for r in universe_pairs):
        lines.append("All configured methods rank **exactly the same** "
                     f"{inspections[0]['n']} `(layer, neuron_index)` pairs.")
    else:
        lines.append("**Universe mismatch.** Pairwise differences:")
        lines.append("")
        lines.append("| A | B | ∩ | only A | only B |")
        lines.append("|---|---|---:|---:|---:|")
        for r in universe_pairs:
            if not r["same"]:
                lines.append(f"| {r['a']} | {r['b']} | {r['intersection']} | "
                             f"{r['only_a']} | {r['only_b']} |")
    lines.append("")

    if identical:
        lines.append("## 3. Identical ranking files")
        lines.append("")
        lines.append("**These methods share a bitwise-identical ranking CSV.**")
        lines.append("They will contribute identical `r_m(j)` values. Their")
        lines.append("top-N eligibility files may still differ.")
        lines.append("")
        for group in identical:
            lines.append(f"- `{', '.join(group)}`")
        lines.append("")

    lines.append("## 4. Ranking prefix vs existing top-N safety-neuron files")
    lines.append("")
    lines.append("For each method and budget N, the ranking prefix of length")
    lines.append("`|top-N file|` is compared as a **set** to that file.")
    lines.append("If the published file is a few neurons off N (ties at the")
    lines.append("cutoff), the file must still be an exact prefix of the ranking.")
    lines.append("")
    lines.append("| Method | N | \\|rank prefix\\| | \\|top-N file\\| | size−N | Jaccard | "
                 "only in ranking | only in top-N | Match |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    mismatches = []
    size_notes = []
    for r in prefix_rows:
        offset = r.get("size_offset", 0) or 0
        if r.get("missing_file"):
            mark = "MISSING"
            mismatches.append(r)
        else:
            mark = "yes" if r["ok"] else "**NO**"
            if not r["ok"]:
                mismatches.append(r)
            elif offset:
                size_notes.append(r)
        lines.append(
            f"| {r['method']} | {r['N']} | {r['rank_n']} | {r['topn_n']} | "
            f"{offset:+d} | {r['jaccard']:.4f} | {r['only_rank']} | "
            f"{r['only_topn']} | {mark} |"
        )
    lines.append("")
    if size_notes:
        lines.append("Published Zhao top-N files are slightly off the nominal")
        lines.append("budget at a few cutoffs (ties). Those files still match")
        lines.append("the ranking prefix of the same length. Eligibility uses")
        lines.append("the published file; the ensemble still emits exactly N")
        lines.append("neurons.")
        lines.append("")
    if mismatches:
        lines.append("### Discrepancies")
        lines.append("")
        lines.append("The methods below **do not** have a ranking prefix that")
        lines.append("reproduces their published top-N safety-neuron file.")
        lines.append("")
        for r in mismatches:
            if r.get("missing_file"):
                lines.append(f"- `{r['method']}` N={r['N']}: missing `{r['path']}`")
            else:
                lines.append(
                    f"- `{r['method']}` N={r['N']}: Jaccard={r['jaccard']:.4f}, "
                    f"{r['only_rank']} neurons in the ranking prefix are absent "
                    f"from the top-N file and {r['only_topn']} published "
                    f"safety neurons are absent from the ranking prefix "
                    f"(`{r['path']}`)."
                )
        lines.append("")
    else:
        lines.append("Every configured method's ranking prefix reproduces its")
        lines.append("published top-N file at every checked budget.")
        lines.append("")

    n_fail = sum(1 for r in inspections if r["issues"])
    n_mismatch = len(mismatches)
    lines.append("## 5. Verdict")
    lines.append("")
    if n_fail == 0 and n_mismatch == 0 and not identical:
        lines.append("PASS: rankings are well-formed, share one universe, and")
        lines.append("agree with the existing top-N safety-neuron files.")
    else:
        bits = []
        if n_fail:
            bits.append(f"{n_fail} ranking-integrity issue(s)")
        if n_mismatch:
            bits.append(f"{n_mismatch} ranking-vs-top-N mismatch(es)")
        if identical:
            bits.append("identical ranking files across methods")
        lines.append("FAIL: " + "; ".join(bits) + ".")
        lines.append("")
        lines.append("Fix the ranking or top-N files before building ensembles.")
    lines.append("")
    return "\n".join(lines)


def run_validation(cfg, budgets, methods=None):
    methods = method_ids(cfg, methods)
    rankings = load_all_rankings(cfg, methods)
    inspections = [inspect_ranking(cfg, mid, rankings[mid]) for mid in methods]
    universe_pairs = compare_universes(inspections)
    prefix_rows = compare_prefix_to_topn(cfg, rankings, budgets, methods)
    identical = identical_ranking_files(inspections)
    ok = (
        all(not r["issues"] for r in inspections)
        and all(r["ok"] for r in prefix_rows)
        and not identical
    )
    report = render_report(cfg, inspections, universe_pairs, prefix_rows, identical, budgets)
    return {
        "ok": ok,
        "report": report,
        "inspections": inspections,
        "universe_pairs": universe_pairs,
        "prefix_rows": prefix_rows,
        "identical_rankings": identical,
        "rankings": rankings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", default=None,
                        help="Write the markdown report here "
                             "(default: rank_ensemble/results/validation_report.md)")
    parser.add_argument("--warn-only", action="store_true",
                        help="Do not exit non-zero on discrepancies.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    budgets = args.budgets or cfg["default_budgets"]
    result = run_validation(cfg, budgets, args.methods)
    out = Path(args.output) if args.output else HERE / "results" / "validation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["report"])
    print(result["report"])
    print(f"\nWrote {out}")
    if not result["ok"] and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

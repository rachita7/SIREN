"""Aggregate ablation result.json files into tables and comparison plots.

    python rank_ensemble/analyze_results.py
    python rank_ensemble/analyze_results.py --budgets 459 2294 4588 9175
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import EXPERIMENTS, display_name, load_config, method_ids, resolve_experiment


def collect_results(results_root, budgets):
    rows = []
    for n in budgets:
        n_dir = Path(results_root) / f"N{n}"
        if not n_dir.exists():
            continue
        for path in sorted(n_dir.glob("target_*/result.json")):
            with open(path) as f:
                r = json.load(f)
            rows.append({
                "budget": n,
                "target_name": r["target_name"],
                "category": r.get("category"),
                "N": r["n_neurons_ablated"],
                "advbench_refusal_rate": r["advbench_refusal_rate"],
                "mmlu_accuracy": r["mmlu_accuracy"],
                "n_advbench_prompts": r.get("n_advbench_prompts"),
                "n_mmlu_questions": r.get("n_mmlu_questions"),
                "timestamp": r.get("timestamp"),
                "path": str(path),
            })
    return pd.DataFrame(rows)


def pretty_name(cfg, target_name):
    if target_name in ("baseline_no_ablation", "unablated", "baseline"):
        return "Unablated"
    if target_name == "rank_consensus":
        return "Rank consensus"
    if target_name == "quota":
        return "Quota ensemble"
    if target_name == "random":
        return "Random"
    if target_name in cfg["method_by_id"]:
        return display_name(cfg, target_name)
    return target_name


def row_order(cfg, methods):
    order = ["baseline_no_ablation", "unablated", "baseline"]
    order += list(methods)
    order += ["rank_consensus", "quota", "random"]
    rank = {name: i for i, name in enumerate(order)}
    return lambda name: rank.get(name, 100 + hash(name) % 100)


def markdown_table(df, cfg):
    lines = [
        "| Selection | N | AdvBench refusal | MMLU accuracy |",
        "| -------------- | ---: | ---------------: | ------------: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {pretty_name(cfg, r['target_name'])} | {int(r['N'])} | "
            f"{r['advbench_refusal_rate']:.3f} | {r['mmlu_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def plot_vs_n(df, cfg, y, ylabel, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    methods = method_ids(cfg)
    styles = {
        "baseline_no_ablation": ("Unablated", "k", "--", 2.0),
        "rank_consensus": ("Rank consensus", "#d62728", "-", 2.4),
        "quota": ("Quota ensemble", "#ff7f0e", "-", 2.4),
        "random": ("Random", "#7f7f7f", ":", 1.8),
    }
    cmap = plt.cm.tab10
    for i, mid in enumerate(methods):
        styles[mid] = (display_name(cfg, mid), cmap(i % 10), "-", 1.3)

    plotted = []
    for name, (label, color, ls, lw) in styles.items():
        sub = df[df["target_name"] == name].sort_values("budget")
        if sub.empty:
            continue
        ax.plot(sub["budget"], sub[y], label=label, color=color,
                linestyle=ls, linewidth=lw, marker="o")
        plotted.append(name)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Neuron budget N")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plane(df, cfg, path, title):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    methods = method_ids(cfg)
    styles = {
        "baseline_no_ablation": ("Unablated", "k", "D", 90),
        "rank_consensus": ("Rank consensus", "#d62728", "s", 80),
        "quota": ("Quota ensemble", "#ff7f0e", "s", 80),
        "random": ("Random", "#7f7f7f", "x", 70),
    }
    cmap = plt.cm.tab10
    for i, mid in enumerate(methods):
        styles[mid] = (display_name(cfg, mid), cmap(i % 10), "o", 55)

    for name, (label, color, marker, size) in styles.items():
        sub = df[df["target_name"] == name]
        if sub.empty:
            continue
        ax.scatter(sub["mmlu_accuracy"], sub["advbench_refusal_rate"],
                   label=label, c=[color], marker=marker, s=size, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(f"N={int(r['budget'])}",
                        (r["mmlu_accuracy"], r["advbench_refusal_rate"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("MMLU accuracy")
    ax.set_ylabel("AdvBench refusal rate")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="all7", choices=list(EXPERIMENTS))
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = resolve_experiment(args.experiment)
    methods = method_ids(cfg, exp["methods"])
    budgets = args.budgets or cfg["default_budgets"]
    results_dir = Path(args.results_dir) if args.results_dir else exp["results_dir"]
    print(f"experiment={exp['name']} results_dir={results_dir}")
    df = collect_results(results_dir, budgets)
    if df.empty:
        raise SystemExit(
            f"No target_*/result.json under {results_dir}/N<budget>/. "
            "Run run_ablation_target.py first."
        )

    key = row_order(cfg, methods)
    parts = []
    md_parts = ["# Rank-ensemble ablation summary", ""]
    md_parts.append(
        "Interpret AdvBench refusal together with MMLU. A near-zero refusal "
        "rate that comes with collapsed MMLU is not a better safety-neuron set."
    )
    md_parts.append("")
    for n in budgets:
        sub = df[df["budget"] == n].copy()
        if sub.empty:
            continue
        sub["_ord"] = sub["target_name"].map(key)
        sub = sub.sort_values(["_ord", "target_name"]).drop(columns="_ord")
        csv_path = results_dir / f"summary_N{n}.csv"
        sub.to_csv(csv_path, index=False)
        parts.append(sub)
        md_parts.append(f"## N = {n}")
        md_parts.append("")
        md_parts.append(markdown_table(sub, cfg))
        md_parts.append("")
        print(f"\nN={n}")
        print(markdown_table(sub, cfg))
        print(f"Wrote {csv_path}")

    all_df = pd.concat(parts, ignore_index=True)
    all_csv = results_dir / "summary_all.csv"
    all_df.to_csv(all_csv, index=False)

    plot_vs_n(df, cfg, "advbench_refusal_rate", "AdvBench refusal rate",
              results_dir / "refusal_vs_N.png")
    plot_vs_n(df, cfg, "mmlu_accuracy", "MMLU accuracy",
              results_dir / "mmlu_vs_N.png")
    plot_plane(df, cfg, results_dir / "refusal_vs_mmlu.png",
               "Ablation: AdvBench refusal vs MMLU (all budgets)")
    for n in budgets:
        sub = df[df["budget"] == n]
        if sub.empty:
            continue
        plot_plane(sub, cfg, results_dir / f"refusal_vs_mmlu_N{n}.png",
                   f"Ablation at N={n}: AdvBench refusal vs MMLU")

    md_path = results_dir / "summary.md"
    md_path.write_text("\n".join(md_parts) + "\n")
    print(f"\nWrote {all_csv}")
    print(f"Wrote {md_path}")
    print(f"Wrote {results_dir}/refusal_vs_N.png, mmlu_vs_N.png, refusal_vs_mmlu*.png")


if __name__ == "__main__":
    main()

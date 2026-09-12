"""Load the rank-ensemble configuration.

Method identities can live in methods.json. That file is optional: the
same defaults are embedded here so a job still runs if methods.json was
not checked in (the repo gitignores *.json).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CONFIG_PATH = HERE / "methods.json"

# Kept in sync with methods.json. Edit either; JSON wins when present.
DEFAULT_RAW = {
    "num_layers": 32,
    "intermediate_size": 14336,
    "default_budgets": [459, 2294, 4588, 9175],
    "eval_seed": 4242,
    "random_seed": 1000,
    "model": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "num_layers": 32,
        "intermediate_size": 14336,
    },
    "methods": [
        {
            "id": "siren",
            "display": "SIREN",
            "ranking": {
                "path": "results/siren_mlpneuron_full_ranking.csv",
                "rank_column": "global_rank",
                "layer_column": "layer",
                "neuron_column": "neuron_index",
                "score_column": "abs_weight",
            },
            "topn": {
                "path": "results/rachita_neurons/llama3-8b-instruct_mlpneuron_mean-std-mlpneuron_mean-clean_selected_neurons_top{n}.json",
                "kind": "layer_json",
            },
        },
        {
            "id": "yang_harmfulness",
            "display": "Yang (harmfulness)",
            "ranking": {
                "path": "results/yang_full_ranking_delta_harmfulness.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "within_layer_index",
                "score_column": "abs_score",
            },
            "topn": {
                "path": "results/tengerleg_neurons/neurons_delta_harmfulness_N{n}.csv",
                "kind": "csv",
            },
        },
        {
            "id": "yang_refusal",
            "display": "Yang (refusal)",
            "ranking": {
                "path": "results/full_ranking_delta_refusal.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "within_layer_index",
                "score_column": "abs_score",
            },
            "topn": {
                "path": "results/tengerleg_neurons/neurons_delta_refusal_N{n}.csv",
                "kind": "csv",
            },
        },
        {
            "id": "zhao_topk",
            "display": "Zhao (top-k)",
            "ranking": {
                "path": "results/full_ranking_neurons_zhao_topk_matching.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "neuron_index",
                "score_column": "score",
            },
            "topn": {
                "path": "results/svea_neurons/fulltest_neurons_zhao_topk_N{n}.csv",
                "kind": "csv",
            },
        },
        {
            "id": "zhao_relative_epsilon",
            "display": "Zhao (rel-eps)",
            "ranking": {
                "path": "results/full_ranking_neurons_zhao_relative_epsilon_matching.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "neuron_index",
                "score_column": "score",
            },
            "topn": {
                "path": "results/svea_neurons/fulltest_neurons_zhao_relative_epsilon_N{n}.csv",
                "kind": "csv",
            },
        },
        {
            "id": "wang",
            "display": "Wang",
            "ranking": {
                "path": "results/full_ranking_neurons_wang_matching.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "neuron_index",
                "score_column": "score",
            },
            "topn": {
                "path": "results/svea_neurons/fulltest_neurons_wang_N{n}.csv",
                "kind": "csv",
            },
        },
        {
            "id": "wang_robust",
            "display": "Wang (robust)",
            "ranking": {
                "path": "results/full_ranking_neurons_wang_robust_matching.csv",
                "rank_column": "rank",
                "layer_column": "layer",
                "neuron_column": "neuron_index",
                "score_column": "score",
            },
            "topn": {
                "path": "results/svea_neurons/fulltest_neurons_wang_robust_N{n}.csv",
                "kind": "csv",
            },
        },
    ],
}


def _read_config_file(path):
    text = Path(path).read_text()
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                f"{path} is YAML but PyYAML is not installed. Use methods.json "
                "or `pip install pyyaml`."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"unsupported config suffix {suffix}; use .json or .yaml")


def load_config(path=None):
    if path is not None:
        path = Path(path)
        raw = _read_config_file(path)
    elif DEFAULT_CONFIG_PATH.exists():
        path = DEFAULT_CONFIG_PATH
        raw = _read_config_file(path)
    else:
        path = DEFAULT_CONFIG_PATH
        raw = DEFAULT_RAW
    if not raw or "methods" not in raw:
        raise ValueError(f"{path} has no methods: list")
    methods = []
    seen = set()
    for spec in raw["methods"]:
        mid = spec["id"]
        if mid in seen:
            raise ValueError(f"duplicate method id {mid!r} in {path}")
        seen.add(mid)
        methods.append(spec)
    num_layers = int(raw.get("num_layers", raw.get("model", {}).get("num_layers", 32)))
    intermediate_size = int(
        raw.get("intermediate_size", raw.get("model", {}).get("intermediate_size", 14336))
    )
    return {
        "path": path,
        "repo_root": REPO_ROOT,
        "here": HERE,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "universe_size": num_layers * intermediate_size,
        "default_budgets": [int(n) for n in raw.get("default_budgets", [459, 2294, 4588, 9175])],
        "eval_seed": int(raw.get("eval_seed", 4242)),
        "random_seed": int(raw.get("random_seed", 1000)),
        "model_name": raw.get("model", {}).get("name", "meta-llama/Meta-Llama-3-8B-Instruct"),
        "methods": methods,
        "method_ids": [m["id"] for m in methods],
        "method_by_id": {m["id"]: m for m in methods},
    }


def method_ids(cfg, only=None):
    ids = list(cfg["method_ids"])
    if only is None:
        return ids
    wanted = list(only)
    unknown = [m for m in wanted if m not in cfg["method_by_id"]]
    if unknown:
        raise KeyError(f"unknown method(s) {unknown}; known: {ids}")
    return wanted


def display_name(cfg, method_id):
    return cfg["method_by_id"][method_id].get("display", method_id)


def resolve(cfg, rel):
    p = Path(rel)
    if p.is_absolute():
        return p
    return cfg["repo_root"] / p

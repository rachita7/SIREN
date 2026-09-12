"""Load the rank-ensemble configuration.

Method identities live in methods.yaml so adding or dropping a method does
not require editing the aggregation or evaluation code.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CONFIG_PATH = HERE / "methods.json"


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
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _read_config_file(path)
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

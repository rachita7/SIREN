"""Build train/val/test splits (65/10/25) for the harmful (HarmBench) and
benign (Alpaca) corpora used across the detection pipeline, and chat-template
every prompt in every split so no downstream script needs to re-derive it.

HarmBench: combines the three raw HF configs already downloaded into
data/full_datasets/harmbench_{standard,contextual,copyright}.csv (400 rows
total) and splits them with category-proportional (stratified) sampling, so
each split gets the same percentage breakdown of categories as the full 400.

Alpaca: samples 400 prompts uniformly at random from the full
data/full_datasets/alpaca_data.json (52002 examples), then splits those 400
with a plain random split (no category field to stratify on).

Both corpora end up with matching split sizes: 260/40/100. Every output row
(train/val/test alike) gets a formatted_input column templated with
meta-llama/Meta-Llama-3-8B-Instruct's own tokenizer -- see format_chat()'s
docstring for the add_special_tokens=False requirement this places on
whatever consumes that column.
"""

import csv
import json
import math
import os
import random
from pathlib import Path

os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.setdefault("HF_HUB_CACHE", f"/work/scratch/{os.environ.get('USER', 'sreuter')}/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"/work/scratch/{os.environ.get('USER', 'sreuter')}/hf_cache/datasets")

from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_DATASETS_DIR = DATA_DIR / "full_datasets"
SEED = 2468
N_SAMPLE = 400
RATIOS = {"train": 0.65, "val": 0.10, "test": 0.25}
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # matches zhao_detection_methods.py / wang_detection_methods.py


def hamilton_apportion(category_sizes: dict, target_total: int) -> dict:
    """Largest-remainder apportionment of target_total across categories,
    proportional to category_sizes. Guarantees the allocations sum to
    exactly target_total (unlike rounding each category independently)."""
    total = sum(category_sizes.values())
    ideal = {k: v * target_total / total for k, v in category_sizes.items()}
    alloc = {k: math.floor(v) for k, v in ideal.items()}
    remainder = target_total - sum(alloc.values())
    order = sorted(category_sizes, key=lambda k: ideal[k] - alloc[k], reverse=True)
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


def stratified_split(rows: list, category_fn, rng: random.Random) -> dict:
    by_category = {}
    for row in rows:
        by_category.setdefault(category_fn(row), []).append(row)

    sizes = {k: len(v) for k, v in by_category.items()}
    test_counts = hamilton_apportion(sizes, round(len(rows) * RATIOS["test"]))
    remaining_sizes = {k: sizes[k] - test_counts[k] for k in sizes}
    val_counts = hamilton_apportion(remaining_sizes, round(len(rows) * RATIOS["val"]))

    splits = {"train": [], "val": [], "test": []}
    for category, items in by_category.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_test = test_counts[category]
        n_val = val_counts[category]
        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])
    return splits


def load_harmbench_rows() -> list:
    rows = []
    with open(FULL_DATASETS_DIR / "harmbench_standard.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"prompt": r["prompt"], "context": "", "category": r["category"], "source_config": "standard"})
    with open(FULL_DATASETS_DIR / "harmbench_contextual.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"prompt": r["prompt"], "context": r["context"], "category": r["category"], "source_config": "contextual"})
    with open(FULL_DATASETS_DIR / "harmbench_copyright.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"prompt": r["prompt"], "context": "", "category": r["tags"], "source_config": "copyright"})
    assert len(rows) == N_SAMPLE, f"expected {N_SAMPLE} HarmBench rows, got {len(rows)}"
    return rows


def build_alpaca_prompt(example: dict) -> str:
    if example["input"].strip():
        return f"{example['instruction']}\n\n{example['input']}"
    return example["instruction"]


def format_chat(tokenizer, prompt: str) -> str:
    """Chat-template a prompt for storage as a CSV string.

    tokenize=False is required here (a CSV cell must be a string, not a
    tensor/dict), unlike zhao_detection_methods.py's format_chat. The
    returned string already contains the literal special tokens (e.g.
    "<|begin_of_text|>"), so whoever tokenizes this column downstream must
    pass add_special_tokens=False -- otherwise the tokenizer inserts a
    second BOS on top of the one already in the string (see
    zhao_detection_methods.py's format_chat docstring for the concrete bug
    this caused previously).
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)

    with open(FULL_DATASETS_DIR / "alpaca_data.json") as f:
        alpaca_data = json.load(f)
    alpaca_sample = rng.sample(alpaca_data, N_SAMPLE)
    alpaca_rows = [
        {
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"],
            "prompt": build_alpaca_prompt(ex),
        }
        for ex in alpaca_sample
    ]

    harmbench_rows = load_harmbench_rows()
    harmbench_splits = stratified_split(harmbench_rows, category_fn=lambda r: r["category"], rng=rng)

    rng.shuffle(alpaca_rows)
    n_test = round(N_SAMPLE * RATIOS["test"])
    n_val = round(N_SAMPLE * RATIOS["val"])
    alpaca_splits = {
        "test": alpaca_rows[:n_test],
        "val": alpaca_rows[n_test : n_test + n_val],
        "train": alpaca_rows[n_test + n_val :],
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split, rows in harmbench_splits.items():
        for row in rows:
            row["formatted_input"] = format_chat(tokenizer, row["prompt"])
        fieldnames = ["prompt", "context", "category", "source_config", "formatted_input"]
        write_csv(DATA_DIR / f"harmbench_{split}.csv", rows, fieldnames)
        print(f"harmbench_{split}.csv: {len(rows)} rows")

    for split, rows in alpaca_splits.items():
        for row in rows:
            row["formatted_input"] = format_chat(tokenizer, row["prompt"])
        fieldnames = ["instruction", "input", "output", "prompt", "formatted_input"]
        write_csv(DATA_DIR / f"alpaca_{split}.csv", rows, fieldnames)
        print(f"alpaca_{split}.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()

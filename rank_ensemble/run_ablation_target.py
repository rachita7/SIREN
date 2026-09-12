"""Ablate one neuron set at a time and score AdvBench + MMLU.

Reproduces Repo B `src/eval/run_ablation_pilot_target.py`:

  * 100 AdvBench prompts, EVAL_SEED=4242, max_new_tokens=64, greedy
  * 2000 MMLU test questions (Repo B CLI defaults to 500; the matched
    experiments used 2000 — see run_zhao_relative_epsilon_ablation.py)
  * refusal_labels.is_refusal
  * next-token logits over " A"/" B"/" C"/" D"
  * result.json per target
  * skip a target if result.json already exists

The only thing that changes across conditions is the neuron set.

    python rank_ensemble/run_ablation_target.py --budget 2294
    python rank_ensemble/run_ablation_target.py --budget 2294 --target-name rank_consensus
    python rank_ensemble/run_ablation_target.py --targets-dir rank_ensemble/results/targets_N2294
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ablation import (
    compute_advbench_refusal_rate,
    load_advbench_prompts,
    load_mmlu_dataset,
    load_model_and_tokenizer,
    now_iso,
    register_ablation_hooks,
    remove_hooks,
    eval_mmlu,
)
from config import HERE as CFG_HERE, load_config
import torch


def load_target(path):
    with open(path) as f:
        return json.load(f)


def run_one_target(model, tokenizer, target, advbench_prompts, mmlu_dataset,
                   max_new_tokens, out_dir, num_layers, intermediate_size,
                   save_raw=False, settings=None):
    name = target["name"]
    flat_idx = torch.tensor(target.get("flat_indices", []), dtype=torch.long)
    target_dir = Path(out_dir) / f"target_{name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / "advbench_raw_generations.jsonl" if save_raw else None

    handles = []
    if flat_idx.numel() > 0:
        handles = register_ablation_hooks(
            model, flat_idx, num_layers, intermediate_size
        )
    try:
        advbench_rate = compute_advbench_refusal_rate(
            model, tokenizer, advbench_prompts, max_new_tokens, raw_path
        )
        mmlu_acc = eval_mmlu(model, tokenizer, mmlu_dataset)
    finally:
        remove_hooks(handles)

    result = {
        "target_name": name,
        "category": target.get("category"),
        "n_neurons_ablated": int(flat_idx.numel()),
        "advbench_refusal_rate": advbench_rate,
        "n_advbench_prompts": len(advbench_prompts),
        "mmlu_accuracy": mmlu_acc,
        "n_mmlu_questions": len(mmlu_dataset),
        "advbench_raw_generations_path": str(raw_path) if raw_path else None,
        "timestamp": now_iso(),
    }
    if settings:
        result["settings"] = settings
    with open(target_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] n_neurons={flat_idx.numel()} "
          f"advbench_refusal_rate={advbench_rate:.3f} "
          f"mmlu_accuracy={mmlu_acc:.3f}")
    return result


def pending_targets(targets_dir, output_dir, target_name=None):
    targets_dir = Path(targets_dir)
    if target_name:
        files = [targets_dir / f"{target_name}.json"]
    else:
        files = sorted(p for p in targets_dir.glob("*.json") if p.name != "manifest.json")
    pending = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        target = load_target(path)
        name = target["name"]
        result_path = Path(output_dir) / f"target_{name}" / "result.json"
        if result_path.exists():
            print(f"[{name}] already done, skipping")
            continue
        pending.append(target)
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--budget", type=int, default=None,
                        help="Use results/targets_N{budget}/ built by build_ensembles.py")
    parser.add_argument("--targets-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--n-advbench-prompts", type=int, default=100)
    parser.add_argument("--n-mmlu-questions", type=int, default=2000,
                        help="Repo B matched experiments used 2000, not the CLI default 500")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--lora-adapter", default=None,
                        help="Optional Peft adapter (Repo B used a DPO LoRA).")
    parser.add_argument("--save-raw-generations", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.eval_seed if args.eval_seed is not None else cfg["eval_seed"]
    if args.targets_dir:
        targets_dir = Path(args.targets_dir)
    elif args.budget is not None:
        targets_dir = CFG_HERE / "results" / f"targets_N{args.budget}"
    else:
        raise SystemExit("pass --budget N or --targets-dir")
    if not targets_dir.exists():
        raise SystemExit(
            f"{targets_dir} does not exist; run "
            f"python rank_ensemble/build_ensembles.py --budgets {args.budget}"
        )

    out_dir = Path(args.output_dir) if args.output_dir else (
        CFG_HERE / "results" / f"N{args.budget}" if args.budget is not None
        else CFG_HERE / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = pending_targets(targets_dir, out_dir, args.target_name)
    if not pending:
        print("nothing to run")
        return

    print(f"loading AdvBench ({args.n_advbench_prompts} prompts) and "
          f"MMLU ({args.n_mmlu_questions} questions)...")
    advbench_prompts = load_advbench_prompts(args.n_advbench_prompts, seed=seed)
    mmlu_dataset = load_mmlu_dataset(args.n_mmlu_questions, seed=seed)

    model_name = args.model or cfg["model_name"]
    print(f"loading model {model_name} ...")
    model, tokenizer = load_model_and_tokenizer(model_name, args.lora_adapter)

    settings = {
        "model": model_name,
        "lora_adapter": args.lora_adapter,
        "eval_seed": seed,
        "n_advbench_prompts": args.n_advbench_prompts,
        "n_mmlu_questions": args.n_mmlu_questions,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "refusal_detector": "rank_ensemble.refusal_labels.is_refusal",
        "mmlu": "cais/mmlu all/test; logits of ' A'/' B'/' C'/' D'",
        "ablation": "forward_pre_hook zeroing mlp.down_proj input channels",
        "source": "Repo B run_ablation_pilot_target.py",
    }
    save_raw_all = args.save_raw_generations is not None and len(args.save_raw_generations) == 0
    save_raw_names = set(args.save_raw_generations or [])

    for target in pending:
        save_raw = save_raw_all or target["name"] in save_raw_names
        run_one_target(
            model, tokenizer, target, advbench_prompts, mmlu_dataset,
            args.max_new_tokens, out_dir, cfg["num_layers"],
            cfg["intermediate_size"], save_raw=save_raw, settings=settings,
        )
    print(f"\ndone -- {len(pending)} target(s) under {out_dir}/target_*/result.json")


if __name__ == "__main__":
    main()

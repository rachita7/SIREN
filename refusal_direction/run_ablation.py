"""Causal validation: does removing each method's neurons actually reduce the
refusal signal and refusal behavior?

The DFA analysis (run_dfa.py) is attributional -- it decomposes what the
neurons write, but writes can be redundant. This script intervenes: for each
method it ZEROES the selected neurons (the down_proj input columns) during
generation on held-out prompts and measures

  1. x . r_hat at the chosen layer's last prompt token -- does the ablation
     drain the residual stream's refusal component?
  2. refusal rate on harmful prompts (substring match on greedy completions)
     -- does behavior change?
  3. refusal rate on benign prompts -- did we merely break the model?

Reference conditions:
  - baseline            no intervention (the numbers everything is read against)
  - direction ablation  project r_hat out of the residual stream at every
                        layer, Arditi et al.'s intervention: the effect size a
                        *complete* removal of the refusal direction produces.
  - random[method]      layer-matched random neurons, the same null as
                        everywhere else in this repo. A method only "matters"
                        if it beats its own random control.

Needs a GPU. Runtime scales with conditions x prompts x max_new_tokens;
defaults (4 methods, 2 null seeds, 100+100 prompts, 64 tokens) take roughly
an hour on a 24 GB card.

Usage:
    python refusal_direction/run_ablation.py \
        --prompts cka/prompts/wildguard.csv \
        --direction refusal_direction/directions/openai_moderation_last.npz
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cka"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuron_sets as ns
import plots
import refusal_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "results")


# ------------------------------------------------------------ interventions

def neuron_ablation_hooks(model, selection, device):
    """Zero the selected down_proj input columns in every forward pass."""
    handles = []
    for layer, idx in selection.items():
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long,
                                device=device)

        def hook(module, args, idx_t=idx_t):
            args[0].index_fill_(-1, idx_t, 0.0)

        handles.append(
            model.model.layers[layer].mlp.down_proj
            .register_forward_pre_hook(hook))
    return handles


def direction_ablation_hooks(model, r_hat, device):
    """Project r_hat out of the residual stream at every write point
    (embedding output and every decoder block output), following Arditi et
    al.'s directional ablation."""
    r = torch.as_tensor(r_hat, dtype=model.dtype, device=device)

    def ablate(h):
        return h - (h @ r).unsqueeze(-1) * r

    handles = [model.model.embed_tokens.register_forward_hook(
        lambda module, args, output: ablate(output))]
    for layer in model.model.layers:
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (ablate(output[0]),) + output[1:]
            return ablate(output)

        handles.append(layer.register_forward_hook(hook))
    return handles


# ------------------------------------------------------------- measurement

@torch.no_grad()
def run_condition(model, tokenizer, texts, r_hat, chosen, device,
                  batch_size, max_new_tokens, max_length):
    """Greedy completions + last-prompt-token projection, current hooks active.

    Prompts are left-padded, so position -1 is always the last real token --
    one padding convention serves both the scoring pass and generation.
    """
    r = torch.as_tensor(r_hat, dtype=torch.float32, device=device)
    completions, projections = [], []
    for start in range(0, len(texts), batch_size):
        chunk = [t if (isinstance(t, str) and t.strip()) else " "
                 for t in texts[start:start + batch_size]]
        batch = tokenizer(chunk, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True,
                          add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch, output_hidden_states=True)
        last = out.hidden_states[chosen][:, -1, :].float()
        projections.extend((last @ r).cpu().numpy().tolist())
        del out

        generated = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        new_tokens = generated[:, batch["input_ids"].shape[1]:]
        completions.extend(
            tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return completions, np.asarray(projections)


def summarize(name, frame):
    rows = {}
    for cls, sub in frame.groupby("class"):
        rows[cls] = {"n": len(sub),
                     "refusal_rate": float(sub["refused"].mean()),
                     "proj_mean": float(sub["proj"].mean()),
                     "proj_std": float(sub["proj"].std(ddof=1))}
    harm = rows.get("harmful", {})
    ben = rows.get("benign", {})
    print(f"  {name:24s} refusal: harmful {harm.get('refusal_rate', float('nan')):.3f} "
          f"benign {ben.get('refusal_rate', float('nan')):.3f}   "
          f"proj(harmful) {harm.get('proj_mean', float('nan')):+8.2f}   "
          f"proj(benign) {ben.get('proj_mean', float('nan')):+8.2f}")
    return {"condition": name,
            "refusal_harmful": harm.get("refusal_rate", float("nan")),
            "refusal_benign": ben.get("refusal_rate", float("nan")),
            "proj_harmful": harm.get("proj_mean", float("nan")),
            "proj_benign": ben.get("proj_mean", float("nan")),
            "n_harmful": harm.get("n", 0), "n_benign": ben.get("n", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", required=True,
                        help="Held-out prompt CSV with formatted_input and "
                             "label columns, e.g. cka/prompts/wildguard.csv. "
                             "Must be disjoint from the direction-fitting set.")
    parser.add_argument("--direction", required=True,
                        help="refusal_direction/directions/{tag}.npz")
    parser.add_argument("--model_path", default=None,
                        help="Defaults to the model recorded in the direction.")
    parser.add_argument("--methods", nargs="+", default=list(ns.DEFAULT_METHODS))
    parser.add_argument("--budget", type=int, default=ns.DEFAULT_BUDGET,
                        choices=ns.BUDGETS)
    parser.add_argument("--n_harmful", type=int, default=100)
    parser.add_argument("--n_benign", type=int, default=100)
    parser.add_argument("--null_seeds", type=int, default=2,
                        help="Layer-matched random draws per method. Each one "
                             "is a full generation pass, so keep this small.")
    parser.add_argument("--skip_direction_ablation", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Enough to catch the refusal prefix.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dir_data = core.load_direction(args.direction)
    chosen = dir_data["chosen"]
    r_hat = dir_data["unit"][chosen]
    model_path = args.model_path or dir_data["model_path"]
    label = args.label or (
        os.path.splitext(os.path.basename(args.prompts))[0]
        + f"_dir-{dir_data['fit_tag']}")
    tag = f"{label}_N{args.budget}"

    frame = pd.read_csv(args.prompts)
    rng = np.random.default_rng(args.seed)
    picked = []
    for cls_label, cls_name, n_want in ((1, "harmful", args.n_harmful),
                                        (0, "benign", args.n_benign)):
        sub = frame[frame["label"] == cls_label]
        take = min(n_want, len(sub))
        pick = sub.iloc[np.sort(rng.choice(len(sub), take, replace=False))].copy()
        pick["class"] = cls_name
        picked.append(pick)
    prompts = pd.concat(picked, ignore_index=True)
    texts = prompts["formatted_input"].astype(str).tolist()
    print(f"{len(prompts)} prompts "
          f"({prompts['class'].value_counts().to_dict()}) | model={model_path}"
          f"\ndirection: hidden state {chosen}, "
          f"val AUROC {dir_data['auroc_val'][chosen]:.4f}")

    selections = {m: ns.load_selection(m, args.budget) for m in args.methods}

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device},
        trust_remote_code=True)
    model.eval()

    # ------------------------------------------------------------ conditions
    conditions = [("baseline", None, None)]
    if not args.skip_direction_ablation:
        conditions.append(("direction ablation", "direction", None))
    for m in args.methods:
        conditions.append((ns.display_name(m), "neurons", selections[m]))
    for m in args.methods:
        for s in range(args.null_seeds):
            rand = ns.random_layer_matched(
                selections[m], np.random.default_rng(args.seed * 1000 + s))
            conditions.append((f"random[{ns.display_name(m)}]#{s}", "neurons",
                               rand))

    print(f"\n{len(conditions)} conditions x {len(prompts)} prompts x "
          f"{args.max_new_tokens} new tokens\n")
    summary_rows, generation_frames = [], []
    for name, kind, payload in conditions:
        handles = []
        if kind == "neurons":
            handles = neuron_ablation_hooks(model, payload, device)
        elif kind == "direction":
            handles = direction_ablation_hooks(model, r_hat, device)
        try:
            completions, projections = run_condition(
                model, tokenizer, texts, r_hat, chosen, device,
                args.batch_size, args.max_new_tokens, args.max_length)
        finally:
            for h in handles:
                h.remove()

        result = prompts[["text", "class", "label"]].copy()
        result["condition"] = name
        result["completion"] = completions
        result["refused"] = [core.is_refusal(c) for c in completions]
        result["proj"] = projections
        generation_frames.append(result)
        summary_rows.append(summarize(name, result))

    # Aggregate each method's random draws into one summary row for the plot.
    plot_rows = [r for r in summary_rows if not r["condition"].startswith("random[")]
    for m in args.methods:
        rand = [r for r in summary_rows
                if r["condition"].startswith(f"random[{ns.display_name(m)}]")]
        if rand:
            plot_rows.append({
                "condition": f"random[{ns.display_name(m)}]",
                **{key: float(np.mean([r[key] for r in rand]))
                   for key in ("refusal_harmful", "refusal_benign",
                               "proj_harmful", "proj_benign")},
                "n_harmful": rand[0]["n_harmful"],
                "n_benign": rand[0]["n_benign"],
            })

    # -------------------------------------------------------------- outputs
    summary_csv = os.path.join(args.output_dir, f"ablation_{tag}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")

    generations_csv = os.path.join(args.output_dir,
                                   f"ablation_generations_{tag}.csv")
    pd.concat(generation_frames, ignore_index=True).to_csv(generations_csv,
                                                           index=False)
    print(f"Saved {generations_csv}  (read a few completions -- substring "
          f"refusal detection is crude)")

    plots.ablation_summary(
        plot_rows,
        os.path.join(args.output_dir, f"ablation_{tag}.png"),
        f"Neuron ablation vs refusal | {label} | N={args.budget}")

    print("\n" + "=" * 72)
    print("How to read this")
    print("=" * 72)
    print("A method matters causally if, RELATIVE TO ITS random[method] "
          "control, ablating\nits neurons (a) lowers proj(harmful) toward the "
          "benign level and (b) lowers the\nharmful refusal rate -- while "
          "refusal on benign prompts stays near baseline\n(otherwise the "
          "ablation just broke the model).")
    print("'direction ablation' shows the ceiling effect a complete removal "
          "of r_hat\nproduces; neuron ablations are diffuse and land "
          "somewhere between baseline\nand that ceiling.")


if __name__ == "__main__":
    main()

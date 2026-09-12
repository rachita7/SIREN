"""Neuron ablation + AdvBench/MMLU evaluation, reproduced from Repo B.

Source mapping (safety-neurons-where-you-look, do not modify that repo):

  register_ablation_hooks / make_zero_input_hook / remove_hooks / format_chat
      src/eval/ablation_validation.py
  load_advbench_prompts / generate_one
      src/eval/ablation_validation_advbench.py
  is_refusal
      src/refusal_labels.py
  load_mmlu_dataset / build_mmlu_prompt / eval_mmlu
      src/eval/run_ablation_pilot_target.py
  random_{n}
      src/build_ablation_pilot_targets.py  (torch.randperm, seed 1000)

Ablation mechanism (Repo B, reproduced here)
--------------------------------------------
A selected FFN neuron is the input channel of `mlp.down_proj` at that layer
— the gated activation SiLU(gate_proj(x)) * up_proj(x), width 14336 for
Llama-3-8B. A `forward_pre_hook` clones that input and zeros the selected
channels, then returns the modified tuple. Equivalently this removes the
neuron's contribution to the residual stream (zeroing the corresponding
down_proj weight column) without changing gate/up computation.

The hook fires on every forward pass, so it applies to every generated
token and to the single-token MMLU logit read. AdvBench and MMLU use the
same hooks.

Repo B registers the hook on
`model.model.model.layers[l].mlp.down_proj` because it wraps the base
model in PeftModel. A plain `LlamaForCausalLM` (the SIREN default) is
`model.model.layers[l].mlp.down_proj`. `decoder_layers()` handles both.

Model default here is `meta-llama/Meta-Llama-3-8B-Instruct` without the
cluster DPO LoRA Repo B hard-codes. Pass `--lora-adapter` to match Repo B's
DPO checkpoint when you have it. Across conditions the model is identical;
only the neuron set changes.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import HERE
from refusal_labels import is_refusal

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ADVBENCH_LOCAL_CACHE = HERE / "data" / "advbench_harmful_behaviors.csv"
MMLU_LETTERS = ["A", "B", "C", "D"]
EVAL_SEED = 4242


def decoder_layers(model):
    """Llama decoder ModuleList, PeftModel or plain CausalLM."""
    if hasattr(model, "peft_config"):
        return model.model.model.layers
    return model.model.layers


def model_device(model):
    """Repo B uses model.device; device_map='auto' sometimes has no .device."""
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    return next(model.parameters()).device


def make_zero_input_hook(neuron_idx_1d):
    """Repo B ablation_validation.make_zero_input_hook.

    Clone down_proj's input and zero the selected FFN channels. The index
    tensor is moved onto the activation device so CPU-built hooks still
    work on GPU.
    """
    def hook(module, args):
        x = args[0]
        idx = neuron_idx_1d
        if idx.device != x.device:
            idx = idx.to(x.device)
        x = x.clone()
        x[..., idx] = 0
        return (x,) + args[1:]
    return hook


def register_ablation_hooks(model, flat_idx, num_layers, intermediate_size):
    """Repo B ablation_validation.register_ablation_hooks.

    flat_idx: 1-D LongTensor of layer * intermediate_size + neuron.
    One forward_pre_hook per affected down_proj.
    """
    by_layer = {}
    for v in flat_idx.tolist():
        layer, neuron = divmod(int(v), int(intermediate_size))
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"flat index {v} maps to layer {layer}, "
                             f"outside 0..{num_layers - 1}")
        by_layer.setdefault(layer, []).append(neuron)

    layers = decoder_layers(model)
    handles = []
    for layer_idx, neurons in by_layer.items():
        idx_tensor = torch.tensor(neurons, dtype=torch.long)
        down_proj = layers[layer_idx].mlp.down_proj
        handles.append(down_proj.register_forward_pre_hook(
            make_zero_input_hook(idx_tensor)
        ))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def format_chat(tokenizer, goal: str) -> str:
    """Repo B ablation_validation.format_chat: user turn + generation prompt."""
    messages = [{"role": "user", "content": goal.strip()}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return goal.strip()


def load_model_and_tokenizer(model_name, lora_adapter=None, attn_implementation="eager"):
    """Repo B load_model_and_tokenizer, without the hardcoded cluster adapter.

    Default: plain Llama-3-8B-Instruct (the model the SIREN selections
    were built on). Optional Peft adapter matches Repo B's DPO setup.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = lora_adapter if lora_adapter else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    if lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, lora_adapter)
    else:
        model = base
    model.eval()
    return model, tokenizer


def load_advbench_prompts(n, seed=EVAL_SEED, cache_path=None):
    """Repo B ablation_validation_advbench.load_advbench_prompts.

    Reads the `goal` column. Fetches the llm-attacks CSV if the local cache
    is missing. Samples `n` prompts with random.seed(seed).
    """
    cache = Path(cache_path) if cache_path else ADVBENCH_LOCAL_CACHE
    if not cache.exists():
        import urllib.request
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ADVBENCH_URL, cache)
    from datasets import load_dataset

    ds = load_dataset("csv", data_files={"train": str(cache)}, split="train")
    if n is None or n < 0 or n >= len(ds):
        return [ds[i]["goal"].strip() for i in range(len(ds))]
    random.seed(seed)
    idx = random.sample(range(len(ds)), n)
    return [ds[i]["goal"].strip() for i in idx]


def build_mmlu_prompt(example):
    """Repo B run_ablation_pilot_target.build_mmlu_prompt. No chat template."""
    lines = [example["question"].strip()]
    for letter, choice in zip(MMLU_LETTERS, example["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_mmlu_dataset(n_questions, seed=EVAL_SEED):
    """Repo B run_ablation_pilot_target.load_mmlu_dataset: cais/mmlu all/test."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    if n_questions is None or n_questions < 0 or n_questions >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n_questions))


@torch.no_grad()
def generate_one(model, tokenizer, goal, max_new_tokens):
    """Repo B ablation_validation_advbench.generate_one: greedy, 64 tokens."""
    prompt_text = format_chat(tokenizer, goal)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device(model))
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


@torch.no_grad()
def eval_mmlu(model, tokenizer, dataset):
    """Repo B run_ablation_pilot_target.eval_mmlu.

    Score the next-token logits of " A"/" B"/" C"/" D" and take argmax.
    Not generation-based multiple choice.
    """
    letter_token_ids = [
        tokenizer(" " + letter, add_special_tokens=False).input_ids[-1]
        for letter in MMLU_LETTERS
    ]
    n_correct = 0
    for example in dataset:
        prompt = build_mmlu_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
        logits = model(**inputs).logits[0, -1]
        pred = int(torch.argmax(logits[letter_token_ids]))
        n_correct += int(pred == example["answer"])
    return n_correct / len(dataset)


@torch.no_grad()
def compute_advbench_refusal_rate(model, tokenizer, prompts, max_new_tokens,
                                  raw_path=None):
    """Repo B run_ablation_pilot_target.compute_advbench_refusal_rate."""
    n_refused = 0
    raw_file = open(raw_path, "w") if raw_path else None
    try:
        for i, goal in enumerate(prompts):
            response = generate_one(model, tokenizer, goal, max_new_tokens)
            refused = is_refusal(response)
            n_refused += int(refused)
            if raw_file:
                raw_file.write(json.dumps({
                    "prompt_index": i,
                    "prompt": goal,
                    "response": response,
                    "is_refusal": refused,
                }) + "\n")
                raw_file.flush()
    finally:
        if raw_file:
            raw_file.close()
    return n_refused / len(prompts) if prompts else float("nan")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

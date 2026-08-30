"""Build held-out evaluation prompt sets for the CKA analysis.

Why held out
------------
All four neuron selections were derived from HarmBench (harmful) + Alpaca
(benign) in data_files/. Measuring representational similarity on those same
prompts would let each method's selection be partly memorized structure of its
own selection set. So this script builds class-balanced prompt sets from
datasets none of the four methods touched.

Recommended pairing (see cka/README.md for the reasoning):
    --dataset wildguard   primary: large, diverse, both classes, prompt-level
                          harm labels, disjoint from HarmBench/Alpaca
    --dataset xstest      stress test: safe prompts that LOOK harmful plus
                          genuinely unsafe ones. Because surface form is
                          decoupled from the label, a high CKA here cannot be
                          explained away as "everyone encodes harmful-looking
                          wording"
Ungated fallbacks if your HF account lacks access to the above:
    --dataset openai_moderation, --dataset beavertails, --dataset aegis2

Formatting
----------
Prompts are wrapped in the Llama-3 chat template with the backbone's own
tokenizer, reproducing the `formatted_input` column of data_files/*.csv, so the
activations sit in the same regime the neurons were selected under. Downstream
extraction must therefore tokenize with add_special_tokens=False.

Output: cka/prompts/{tag}.csv with columns
    text, formatted_input, label, dataset
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "prompts")

# Direct loaders rather than train/preprocess.py, for two reasons that matter
# here:
#   1. preprocess.py splits every corpus into train/val/test for probe
#      training. None of these corpora were used to select any method's
#      neurons, so the WHOLE dataset is held out and splitting it just throws
#      prompts away -- for XSTest (450 rows) the test split is 90 prompts,
#      far too few for CKA.
#   2. preprocess.py's wildguard path loads `wildguardtrain` (~87k rows) as
#      well as `wildguardtest`, which is a large download of a gated config we
#      have no use for.
# `HF gated?` marks datasets that need terms accepted on the Hub plus a token.


def _clean(frame):
    frame = frame[["text", "label"]].dropna()
    frame = frame.copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0]
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def _wildguard():
    """WildGuard test split: prompt-level harm labels, adversarial + vanilla.

    HF gated: accept the terms at
    https://huggingface.co/datasets/allenai/wildguardmix
    """
    from datasets import load_dataset

    frame = load_dataset("allenai/wildguardmix", "wildguardtest")["test"].to_pandas()
    frame = frame[frame["prompt_harm_label"].isin(["harmful", "unharmful"])]
    frame["text"] = frame["prompt"]
    frame["label"] = (frame["prompt_harm_label"] == "harmful").astype(int)
    return _clean(frame)


def _xstest():
    """All 450 XSTest prompts: 250 safe-but-harmful-looking + 200 unsafe.

    The whole corpus is used deliberately -- this set's value is that surface
    form is decoupled from the label, and it is small enough that discarding
    any of it would leave too few prompts.
    """
    from datasets import load_dataset

    frame = load_dataset("Paul/XSTest")["train"].to_pandas()
    frame["text"] = frame["prompt"]
    frame["label"] = (frame["label"].astype(str) == "unsafe").astype(int)
    return _clean(frame)


def _aegis2():
    """Aegis 2.0 test split (1,964 rows). HF gated."""
    from datasets import load_dataset

    frame = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0")["test"].to_pandas()
    frame = frame[frame["prompt_label"].isin(["safe", "unsafe"])]
    frame["text"] = frame["prompt"]
    frame["label"] = (frame["prompt_label"] == "unsafe").astype(int)
    return _clean(frame)


def _openai_moderation():
    """OpenAI moderation set (1,680 rows), ungated. Harmful = any category flagged."""
    from datasets import load_dataset

    frame = load_dataset("walledai/openai-moderation-dataset")["train"].to_pandas()
    cols = [c for c in ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")
            if c in frame.columns]
    frame["text"] = frame["prompt"]
    frame["label"] = (frame[cols].astype(int).sum(axis=1) > 0).astype(int)
    return _clean(frame)


def _toxic_chat():
    """ToxicChat test split. HF gated."""
    from datasets import load_dataset

    frame = load_dataset("lmsys/toxic-chat", "toxicchat0124")["test"].to_pandas()
    frame["text"] = frame["user_input"]
    frame["label"] = frame["toxicity"].astype(int)
    return _clean(frame)


def _beavertails():
    """BeaverTails 30k test split, ungated.

    Its labels describe the RESPONSE, so text is prompt + response (matching
    train/preprocess.py). Use it as an ungated fallback, keeping in mind it
    measures response-level rather than prompt-level harm.
    """
    from datasets import load_dataset

    frame = load_dataset("PKU-Alignment/BeaverTails")["30k_test"].to_pandas()
    frame["text"] = frame["prompt"].astype(str) + "\n" + frame["response"].astype(str)
    frame["label"] = (~frame["is_safe"].astype(bool)).astype(int)
    return _clean(frame)


def _advbench():
    """AdvBench, ungated. HARMFUL ONLY -- no class-residualized variant."""
    from datasets import load_dataset

    frame = load_dataset("walledai/AdvBench")["train"].to_pandas()
    frame["text"] = frame["prompt"]
    frame["label"] = 1
    return _clean(frame)


LOADERS = {
    "wildguard": _wildguard,
    "xstest": _xstest,
    "aegis2": _aegis2,
    "openai_moderation": _openai_moderation,
    "toxic_chat": _toxic_chat,
    "beavertails": _beavertails,
    "advbench": _advbench,
}
GATED = {"wildguard", "aegis2", "toxic_chat"}
SUPPORTED = LOADERS


def load_texts(dataset):
    try:
        return LOADERS[dataset]()
    except Exception as exc:
        if dataset in GATED:
            raise SystemExit(
                f"Failed to load '{dataset}': {exc}\n\n"
                f"'{dataset}' is gated on HuggingFace. To use it:\n"
                f"  1. Accept the terms on the dataset's Hub page while logged in.\n"
                f"  2. Authenticate:  huggingface-cli login   (or export HF_TOKEN=hf_...)\n"
                f"Ungated alternatives that need no access request: "
                f"{', '.join(sorted(set(LOADERS) - GATED))}")
        raise


def deduplicate(frame):
    """Drop repeated prompts.

    Duplicates are actively harmful for CKA, not merely wasteful: two identical
    prompts produce two identical rows, hence a block of maximal similarity in
    every method's prompt-similarity matrix. That block is shared by all methods
    regardless of which neurons they selected, so it inflates every CKA. Matters
    most when pooling datasets, which overlap in sources.
    """
    before = len(frame)
    frame = frame.drop_duplicates(subset="text").reset_index(drop=True)
    if len(frame) < before:
        print(f"  dropped {before - len(frame)} duplicate prompts")
    return frame


def balance(frame, max_prompts, seed):
    """Equal numbers of harmful and benign prompts, capped at max_prompts.

    Balancing matters here beyond the usual reasons: the class-residualized
    CKA subtracts per-class means, and an unbalanced set would make one class's
    mean far noisier than the other's.

    The binding constraint is usually the smaller class, not max_prompts.
    WildGuard's test split, for instance, is 1,725 prompts but only 754 are
    harmful, so a balanced set is 1,508 however high max_prompts is. Pool
    several datasets to get past that.
    """
    rng = np.random.default_rng(seed)
    groups = {int(v): sub for v, sub in frame.groupby("label")}
    if len(groups) < 2:
        print(f"  WARNING: only class {list(groups)} present -- the "
              f"class-residualized CKA variant will be uninformative here.")
        per_class = max_prompts
    else:
        smallest = min(len(g) for g in groups.values())
        per_class = min(smallest, max_prompts // 2)
        if smallest < max_prompts // 2:
            print(f"  NOTE: the smaller class has {smallest} prompts, so the "
                  f"balanced set is {2 * smallest}, below the requested "
                  f"{max_prompts}. Pass several datasets to --dataset to pool "
                  f"them and reach the target.")
    picked = []
    for label in sorted(groups):
        sub = groups[label]
        take = min(per_class, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        picked.append(sub.iloc[np.sort(idx)])
    out = pd.concat(picked, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def apply_template(texts, model_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return [
        tok.apply_chat_template([{"role": "user", "content": t}],
                                tokenize=False, add_generation_prompt=True)
        for t in texts
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build a held-out, class-balanced, chat-templated prompt set.")
    parser.add_argument("--dataset", required=True, nargs="+",
                        choices=sorted(SUPPORTED),
                        help="One or more datasets. Several are POOLED into a "
                             "single balanced prompt set, which is how you get "
                             "past a small dataset's minority-class limit. The "
                             "`dataset` column is preserved so run_cka.py can "
                             "project out dataset identity.")
    parser.add_argument("--max_prompts", type=int, default=2000,
                        help="Total prompts (half per class). >=2000 is "
                             "advisable: with a 2294-neuron budget, fewer "
                             "prompts than neurons makes the prompt-similarity "
                             "matrix rank-deficient and inflates biased CKA.")
    parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Tokenizer used for the chat template; must match "
                             "the backbone the neurons were selected on.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tag", default=None,
                        help="Output basename; defaults to the dataset name.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tag = args.tag or "+".join(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    frames = []
    for name in args.dataset:
        print(f"Loading {name}"
              + ("  (gated on HuggingFace)" if name in GATED else "") + " ...")
        sub = load_texts(name)
        sub["dataset"] = name
        print(f"  {len(sub)} usable prompts, "
              f"class balance {sub['label'].value_counts().to_dict()}")
        frames.append(sub)

    frame = pd.concat(frames, ignore_index=True)
    frame = deduplicate(frame)
    if len(args.dataset) > 1:
        print(f"  pooled: {len(frame)} prompts, "
              f"class balance {frame['label'].value_counts().to_dict()}")

    frame = balance(frame, args.max_prompts, args.seed)
    print(f"  kept {len(frame)}, balanced to "
          f"{frame['label'].value_counts().to_dict()}")
    if len(args.dataset) > 1:
        print(f"  per dataset: {frame['dataset'].value_counts().to_dict()}")

    print(f"Applying the {args.model_path} chat template ...")
    frame["formatted_input"] = apply_template(frame["text"].tolist(),
                                              args.model_path)

    path = os.path.join(args.output_dir, f"{tag}.csv")
    frame[["text", "formatted_input", "label", "dataset"]].to_csv(path, index=False)
    print(f"Saved {path}")
    print("\nNext: cka/extract_activations.py --prompts "
          f"{os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()

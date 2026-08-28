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

# Datasets already implemented by train/preprocess.py, all returning
# DatasetDict entries with `text` and binary `label` (1 = harmful).
SUPPORTED = {
    "wildguard": "test",
    "xstest": "test",
    "aegis2": "test",
    "openai_moderation": "test",
    "toxic_chat": "test",
    "beavertails": "test",
    "safe_rlhf": "test",
    "advbench": "test",
}


def load_texts(dataset, split):
    from train.preprocess import preprocess_dataset

    ds = preprocess_dataset(dataset)
    if split not in ds:
        raise KeyError(f"{dataset} has splits {list(ds)}, not '{split}'")
    frame = ds[split].to_pandas()
    if "text" not in frame or "label" not in frame:
        raise ValueError(f"{dataset}/{split} lacks text/label columns: "
                         f"{list(frame.columns)}")
    frame = frame[["text", "label"]].dropna()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0]
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def balance(frame, max_prompts, seed):
    """Equal numbers of harmful and benign prompts, capped at max_prompts.

    Balancing matters here beyond the usual reasons: the class-residualized
    CKA subtracts per-class means, and an unbalanced set would make one class's
    mean far noisier than the other's.
    """
    rng = np.random.default_rng(seed)
    groups = {int(v): sub for v, sub in frame.groupby("label")}
    if len(groups) < 2:
        print(f"  WARNING: only class {list(groups)} present -- the "
              f"class-residualized CKA variant will be uninformative here.")
        per_class = max_prompts
    else:
        per_class = min(min(len(g) for g in groups.values()), max_prompts // 2)
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
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--split", default=None,
                        help="Default: the dataset's test split.")
    parser.add_argument("--max_prompts", type=int, default=2000,
                        help="Total prompts (half per class). >=2000 is "
                             "advisable: with a 2500-neuron budget, fewer "
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

    split = args.split or SUPPORTED[args.dataset]
    tag = args.tag or args.dataset
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset}/{split} ...")
    frame = load_texts(args.dataset, split)
    print(f"  {len(frame)} usable prompts, "
          f"class balance {frame['label'].value_counts().to_dict()}")

    frame = balance(frame, args.max_prompts, args.seed)
    print(f"  kept {len(frame)}, balanced to "
          f"{frame['label'].value_counts().to_dict()}")

    print(f"Applying the {args.model_path} chat template ...")
    frame["formatted_input"] = apply_template(frame["text"].tolist(),
                                              args.model_path)
    frame["dataset"] = args.dataset

    path = os.path.join(args.output_dir, f"{tag}.csv")
    frame[["text", "formatted_input", "label", "dataset"]].to_csv(path, index=False)
    print(f"Saved {path}")
    print("\nNext: cka/extract_activations.py --prompts "
          f"{os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()

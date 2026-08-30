"""Build the prompt sets used to FIT and VALIDATE the refusal direction.

The study uses ONE dataset framework, WildGuard, with the official splits
doing the train/test separation (our controlled adaptation of Arditi et
al.'s train -> validation -> evaluation protocol):

  WildGuardTrain -> fit subset    construct the difference-in-means direction
  WildGuardTrain -> val subset    choose the direction's layer (AUROC)
  WildGuardTest  (cka/prompts/wildguard.csv)   everything that is REPORTED:
                                  CKA, DFA, write-vector cosines, ablation

The one principle that matters: prompts used to construct/select the
direction must not be the prompts used for final evaluation. The official
train/test split provides that, and as a belt-and-braces guard any text that
also appears in the evaluation prompt CSVs is dropped before balancing.
(The four neuron sets themselves were selected earlier on HarmBench +
Alpaca -- upstream provenance, not part of this experiment.)

The direction is a difference-in-means, so it needs both classes but not
many prompts (Arditi et al. fit theirs on 128 per class). The fit/val split
is stratified; selecting the layer on the fitting prompts would overfit the
depth choice.

Output: refusal_direction/prompts/{tag}_fit.csv and {tag}_val.csv with
columns text, formatted_input, label, dataset (same schema as cka/prompts).
"""
import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cka"))

import numpy as np
import pandas as pd

# Reuse the CKA folder's dataset loaders, dedup/balance logic and chat
# templating so the two analyses cannot drift apart in preprocessing.
from build_prompts import GATED, LOADERS, _clean, apply_template, balance, \
    deduplicate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "prompts")
DEFAULT_EXCLUDE_GLOB = os.path.join(REPO_ROOT, "cka", "prompts", "*.csv")


def _wildguard_train():
    """WildGuardTrain (~87k rows): prompt-level harm labels, both classes.

    The official TRAIN split of the same gated dataset whose TEST split is
    the evaluation set (cka/prompts/wildguard.csv), so the train/test
    separation is WildGuard's own, not something we invented. Labels here
    are GPT-4-derived with auditing (the test split is human-annotated),
    which is fine for fitting a mean-difference direction. Deliberately NOT
    added to cka/build_prompts.py: the CKA analysis must never evaluate on
    train-split prompts.
    """
    from datasets import load_dataset

    frame = load_dataset("allenai/wildguardmix",
                         "wildguardtrain")["train"].to_pandas()
    frame = frame[frame["prompt_harm_label"].isin(["harmful", "unharmful"])]
    frame["text"] = frame["prompt"]
    frame["label"] = (frame["prompt_harm_label"] == "harmful").astype(int)
    return _clean(frame)


DIRECTION_LOADERS = dict(LOADERS, wildguard_train=_wildguard_train)
DIRECTION_GATED = GATED | {"wildguard_train"}


def drop_overlap(frame, exclude_paths):
    """Remove any prompt whose text appears in one of the excluded CSVs."""
    seen = set()
    for path in exclude_paths:
        try:
            other = pd.read_csv(path, usecols=["text"])
        except Exception as exc:
            print(f"  WARNING: could not read exclude file {path}: {exc}")
            continue
        seen.update(other["text"].astype(str).str.strip())
    if not seen:
        return frame
    mask = ~frame["text"].astype(str).str.strip().isin(seen)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  dropped {dropped} prompts that also appear in the "
              f"evaluation sets ({len(exclude_paths)} files checked)")
    return frame[mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build disjoint fit/val prompt sets for the refusal direction.")
    parser.add_argument("--dataset", nargs="+", default=["wildguard_train"],
                        choices=sorted(DIRECTION_LOADERS),
                        help="Direction-fitting corpora (pooled). The default "
                             "wildguard_train is the official TRAIN split of "
                             "the evaluation dataset; any other choice must "
                             "stay disjoint from the evaluation prompts "
                             "(enforced via --exclude anyway).")
    parser.add_argument("--max_prompts", type=int, default=1024,
                        help="Total after balancing (half per class). A "
                             "difference-in-means needs far fewer prompts "
                             "than CKA; 512 per class is already generous.")
    parser.add_argument("--val_fraction", type=float, default=0.25,
                        help="Share held out for layer selection.")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="CSVs whose 'text' rows must not appear here. "
                             f"Default: every file matching {DEFAULT_EXCLUDE_GLOB}")
    parser.add_argument("--model_path",
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tag", default=None,
                        help="Output basename; defaults to the dataset name.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tag = args.tag or "+".join(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    exclude = (args.exclude if args.exclude is not None
               else sorted(glob.glob(DEFAULT_EXCLUDE_GLOB)))

    frames = []
    for name in args.dataset:
        print(f"Loading {name}"
              + ("  (gated on HuggingFace)" if name in DIRECTION_GATED else "")
              + " ...")
        sub = DIRECTION_LOADERS[name]()
        sub["dataset"] = name
        print(f"  {len(sub)} usable prompts, "
              f"class balance {sub['label'].value_counts().to_dict()}")
        frames.append(sub)

    frame = pd.concat(frames, ignore_index=True)
    frame = deduplicate(frame)
    frame = drop_overlap(frame, exclude)
    frame = balance(frame, args.max_prompts, args.seed)
    print(f"  kept {len(frame)}, balanced to "
          f"{frame['label'].value_counts().to_dict()}")

    # Stratified fit/val split so both classes appear in both halves.
    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(frame), dtype=bool)
    for label, sub in frame.groupby("label"):
        n_val = max(1, int(round(len(sub) * args.val_fraction)))
        pick = rng.choice(sub.index.to_numpy(), size=n_val, replace=False)
        val_mask[pick] = True

    print(f"Applying the {args.model_path} chat template ...")
    frame["formatted_input"] = apply_template(frame["text"].tolist(),
                                              args.model_path)

    cols = ["text", "formatted_input", "label", "dataset"]
    for split, sub in (("fit", frame[~val_mask]), ("val", frame[val_mask])):
        path = os.path.join(args.output_dir, f"{tag}_{split}.csv")
        sub[cols].to_csv(path, index=False)
        print(f"Saved {path}  ({len(sub)} prompts, "
              f"{sub['label'].value_counts().to_dict()})")

    print("\nNext: refusal_direction/extract_residuals.py --prompts "
          f"refusal_direction/prompts/{tag}_fit.csv  (and _val.csv)")


if __name__ == "__main__":
    main()

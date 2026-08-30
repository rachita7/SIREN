"""Fit the refusal direction (difference-in-means) and select its layer.

For every hidden-state index h the direction is

    r^(h) = mean over harmful fit prompts of x_h  -  mean over harmless ones

computed at the last prompt token. The h actually used downstream is chosen
on the VALIDATION split by projection AUROC -- how well x . r_hat separates
held-out harmful from harmless prompts -- restricted to the first 80% of
depth (Arditi et al. exclude late layers, whose directions are entangled
with specific output tokens).

Output:
  refusal_direction/directions/{tag}.npz   directions [H, D] + selection info
  refusal_direction/results/direction_profile_{tag}.png

CPU-only; the GPU work happened in extract_residuals.py.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import plots
import refusal_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRECTIONS_DIR = os.path.join(HERE, "directions")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")


def load_residuals(path):
    meta_path = os.path.splitext(path)[0] + ".meta.csv"
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing metadata file {meta_path}")
    resid = np.load(path)
    meta = pd.read_csv(meta_path)
    if len(meta) != resid.shape[0]:
        raise SystemExit(f"{path} has {resid.shape[0]} rows but "
                         f"{meta_path} has {len(meta)}")
    return resid, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit_residuals", required=True,
                        help="npy from extract_residuals.py on the FIT split.")
    parser.add_argument("--val_residuals", required=True,
                        help="Same, on the VAL split (layer selection).")
    parser.add_argument("--max_frac", type=float, default=0.8,
                        help="Hidden states beyond this depth fraction are "
                             "excluded from selection.")
    parser.add_argument("--model_path",
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Recorded in the artifact so run_dfa.py can "
                             "check it matches the weights it streams.")
    parser.add_argument("--tag", default=None,
                        help="Artifact basename; defaults to the fit file's.")
    parser.add_argument("--directions_dir", default=DEFAULT_DIRECTIONS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    tag = args.tag or os.path.splitext(
        os.path.basename(args.fit_residuals))[0].replace("_fit", "")
    os.makedirs(args.directions_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    fit_resid, fit_meta = load_residuals(args.fit_residuals)
    val_resid, val_meta = load_residuals(args.val_residuals)
    n_fit, num_hidden, d_model = fit_resid.shape
    print(f"Fit: {n_fit} prompts | val: {val_resid.shape[0]} prompts | "
          f"{num_hidden} hidden states x {d_model} dims")
    print(f"  fit classes: {fit_meta['label'].value_counts().to_dict()}"
          f"   val classes: {val_meta['label'].value_counts().to_dict()}")

    directions = core.diff_in_means(fit_resid, fit_meta["label"].to_numpy())
    units = core.unit(directions, axis=1)

    val_labels = val_meta["label"].to_numpy()
    aurocs = np.full(num_hidden, np.nan)
    ds = np.full(num_hidden, np.nan)
    for h in range(num_hidden):
        scores = val_resid[:, h].astype(np.float32) @ units[h]
        aurocs[h] = core.auroc(scores, val_labels)
        ds[h] = core.cohens_d(scores, val_labels)

    chosen = core.choose_hidden(aurocs, max_frac=args.max_frac)
    num_layers = num_hidden - 1

    print(f"\nValidation AUROC of the projection x . r_hat per hidden state")
    print(f"  (hidden state h = output of decoder block h-1; h=0 = embeddings)")
    for h in range(num_hidden):
        marker = "  <-- chosen" if h == chosen else ""
        capped = "  (beyond depth cap)" if h >= int(np.ceil(
            args.max_frac * num_hidden)) else ""
        print(f"  h={h:2d}  AUROC={aurocs[h]:.4f}  d={ds[h]:+.2f}  "
              f"|r|={np.linalg.norm(directions[h]):8.3f}{marker}{capped}")
    print(f"\nChosen hidden state: {chosen} (output of block {chosen - 1}), "
          f"val AUROC {aurocs[chosen]:.4f}, Cohen's d {ds[chosen]:+.2f}")

    out_path = os.path.join(args.directions_dir, f"{tag}.npz")
    core.save_direction(out_path, directions, aurocs, ds, chosen,
                        args.model_path, tag, num_layers)
    print(f"Saved {out_path}")

    plots.direction_profile(
        aurocs, ds, chosen, args.max_frac,
        os.path.join(args.output_dir, f"direction_profile_{tag}.png"),
        f"Refusal-direction separation by depth | {tag}")


if __name__ == "__main__":
    main()
